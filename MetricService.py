import numpy as np
from pydantic import BaseModel
from scipy import stats
import pandas as pd

from datetime import datetime

class Design(BaseModel):
    """Дата-класс с описание параметров эксперимента.

    statistical_test - тип статтеста. ['ttest', 'bootstrap']
    effect - размер эффекта в процентах
    alpha - уровень значимости
    beta - допустимая вероятность ошибки II рода
    bootstrap_iter - количество итераций бутстрепа
    bootstrap_ci_type - способ построения доверительного интервала. ['normal', 'percentile', 'pivotal']
    bootstrap_agg_func - метрика эксперимента. ['mean', 'quantile 95']
    metric_name - название целевой метрики эксперимента
    metric_outlier_lower_bound - нижняя допустимая граница метрики, всё что ниже считаем выбросами
    metric_outlier_upper_bound - верхняя допустимая граница метрики, всё что выше считаем выбросами
    metric_outlier_process_type - способ обработки выбросов. ['drop', 'clip'].
        'drop' - удаляем измерение, 'clip' - заменяем выброс на значение ближайшей границы (lower_bound, upper_bound).
    """
    statistical_test: str = 'ttest'
    effect: float = 3.
    alpha: float = 0.05
    beta: float = 0.1
    bootstrap_iter: int = 1000
    bootstrap_ci_type: str = 'normal'
    bootstrap_agg_func: str = 'mean'
    metric_name: str
    metric_outlier_lower_bound: float
    metric_outlier_upper_bound: float
    metric_outlier_process_type: str

class MetricsService:

    def __init__(self, data_service):
        """Класс для вычисления метрик.

        :param data_service (DataService): объект класса, предоставляющий доступ с данным.
        """
        self.data_service = data_service

    def _get_data_subset(self, table_name, begin_date, end_date, user_ids=None, columns=None):
        """Возвращает часть таблицы с данными."""
        return self.data_service.get_data_subset(table_name, begin_date, end_date, user_ids, columns)

    def _calculate_revenue_web(self, begin_date, end_date, user_ids):
        """Вычисляет метрику суммарная выручка с пользователя за указанный период
        для заходивших на сайт в указанный период.

        Эта метрика нужна для экспериментов на сайте, когда в эксперимент попадают только те, кто заходил на сайт.

        Нужно вернуть значения user_id и суммарной выручки (sum(price)).
        Данный о ценах в таблице 'sales'. Данные о заходивших на сайт в таблице 'web-logs'.
        Если пользователь зашёл на сайт и ничего не купил, его суммарная стоимость покупок равна нулю.
        Для каждого user_id должно быть ровно одно значение.

        :param begin_date, end_date (datetime): период времени, за который нужно считать метрику.
            Также за этот период времени нужно выбирать пользователей, которые заходили на сайт.
        :param user_id (None, list[str]): id пользователей, по которым нужно отфильтровать полученные значения.

        :return (pd.DataFrame): датафрейм с двумя столбцами ['user_id', 'metric']
        """
        user_ids_ = (
            self._get_data_subset('web-logs', begin_date, end_date, user_ids, ['user_id'])
            ['user_id'].unique()
        )
        df = (
            self._get_data_subset('sales', begin_date, end_date, user_ids, ['user_id', 'price'])
            .groupby('user_id')[['price']].sum().reset_index()
            .rename(columns={'price': 'metric'})
        )
        df = pd.merge(pd.DataFrame({'user_id': user_ids_}), df, on='user_id', how='left').fillna(0)
        return df[['user_id', 'metric']]

    @staticmethod
    def _calculate_cuped_theta(y, y_cov) -> float:
        """Вычисляем Theta.

        y - значения метрики во время пилота
        y_cov - значения ковариант
        """
        covariance = np.cov(y_cov, y)[0, 1]
        variance = y_cov.var()
        theta = covariance / variance
        return theta

    def _calculate_cuped_metric(self, df) -> pd.Series:
        """Вычисляем столбец с CUPED метрикой.

        :param df: датафрейм с метриками, columns = ['user_id', 'metric', 'metric_cov']
        """
        y = df['metric'].values
        y_cov = df['metric_cov'].values
        theta = self._calculate_cuped_theta(y, y_cov)
        return df['metric'] - theta * (df['metric_cov'] - df['metric_cov'].mean())

    def calculate_metric(self, metric_name, begin_date, end_date, cuped, user_ids=None):
        """Считает значения метрики.

        :param metric_name (str): название метрики
        :param begin_date (datetime): дата начала периода (включая границу)
        :param end_date (datetime): дата окончания периода (не включая границу)
        :param cuped (str): применение CUPED. ['off', 'on (previous week revenue)']
            'off' - не применять CUPED
            'on (previous week revenue)' - применяем CUPED, в качестве ковариаты
                используем выручку за прошлую неделю
        :param user_ids (list[str], None): список пользователей.
            Если None, то вычисляет метрику для всех пользователей.
        :return df: columns=['user_id', 'metric']
        """
        if metric_name == 'revenue (web)':
            if cuped == 'off':
                return self._calculate_revenue_web(begin_date, end_date, user_ids)
            elif cuped == 'on (previous week revenue)':
                df = self._calculate_revenue_web(begin_date, end_date, user_ids)
                begin_date_cov, end_date_cov = begin_date - timedelta(days=7), begin_date
                df_cov = (
                    self._calculate_revenue_web(begin_date_cov, end_date_cov, user_ids)
                    .rename(columns={'metric': 'metric_cov'})
                )
                df = pd.merge(df, df_cov, on='user_id', how='left').fillna(0)
                df['metric'] = self._calculate_cuped_metric(df)
                return df[['user_id', 'metric']]
            else:
                raise ValueError('Wrong cuped')
        else:
            raise ValueError('Wrong metric name')

    def process_outliers(self, metrics, design):
        """Возвращает новый датафрейм с обработанными выбросами в измерениях метрики.

        :param metrics (pd.DataFrame): таблица со значениями метрики, columns=['user_id', 'metric'].
        :param design (Design): объект с данными, описывающий параметры эксперимента.
        :return df: columns=['user_id', 'metric']
        """
        lower_bound = design.metric_outlier_lower_bound
        upper_bound = design.metric_outlier_upper_bound
        metrics = metrics.copy()
        if design.metric_outlier_process_type == 'drop':
            metrics = metrics[(metrics['metric'] >= lower_bound) & (metrics['metric'] <= upper_bound)]
        elif design.metric_outlier_process_type == 'clip':
            metrics.loc[metrics['metric'] < lower_bound, 'metric'] = lower_bound
            metrics.loc[metrics['metric'] > upper_bound, 'metric'] = upper_bound
        else:
            raise ValueError('Неверное значение design.metric_outlier_process_type')
        return metrics

    def calculate_linearized_metrics(
            self, control_metrics, pilot_metrics, control_user_ids=None, pilot_user_ids=None
    ):
        """Считает значения метрики отношения.

        Нужно вычислить параметр kappa (коэффициент в функции линеаризации) по данным из
        control_metrics и использовать его для вычисления линеаризованной метрики.

        :param control_metrics (pd.DataFrame): датафрейм со значениями метрики контрольной группы.
            Значения в столбце 'user_id' не уникальны.
            Измерения для одного user_id считаем зависимыми, а разных user_id - независимыми.
            columns=['user_id', 'metric']
        :param pilot_metrics (pd.DataFrame): датафрейм со значениями метрики экспериментальной группы.
            Значения в столбце 'user_id' не уникальны.
            Измерения для одного user_id считаем зависимыми, а разных user_id - независимыми.
            columns=['user_id', 'metric']
        :param control_user_ids (list): список id пользователей контрольной группы, для которых
            нужно рассчитать метрику. Если None, то использовать пользователей из control_metrics.
            Если для какого-то пользователя нет записей в таблице control_metrics, то его
            линеаризованная метрика равна нулю.
        :param pilot_user_ids (list): список id пользователей экспериментальной группы, для которых
            нужно рассчитать метрику. Если None, то использовать пользователей из pilot_metrics.
            Если для какого-то пользователя нет записей в таблице pilot_metrics, то его
            линеаризованная метрика равна нулю.
        :return lin_control_metrics, lin_pilot_metrics: columns=['user_id', 'metric']
        """
        kappa = control_metrics['metric'].mean()

        dfs = []
        for df, user_ids in [(control_metrics, control_user_ids,), (pilot_metrics, pilot_user_ids,)]:
            df_agg = df.groupby('user_id')[['metric']].agg(['sum', 'count'])
            df_agg.columns = df_agg.columns.get_level_values(1)
            df_agg['metric'] = df_agg['sum'] - kappa * df_agg['count']

            if user_ids:
                df_user = pd.DataFrame({'user_id': user_ids})
            else:
                df_user = pd.DataFrame({'user_id': df_agg.index.values})
            df_res = pd.merge(
                df_user,
                df_agg[['metric']].reset_index(),
                on='user_id',
                how='left'
            ).fillna(0)
            dfs.append(df_res)
        return dfs


    def _chech_df(df, df_ideal, sort_by, reindex=False, set_dtypes=False):
        assert isinstance(df, pd.DataFrame), 'Функция вернула не pd.DataFrame.'
        assert len(df) == len(df_ideal), 'Неверное количество строк.'
        assert len(df.T) == len(df_ideal.T), 'Неверное количество столбцов.'
        columns = df_ideal.columns
        assert df.columns.isin(columns).sum() == len(df.columns), 'Неверное название столбцов.'
        df = df[columns].sort_values(sort_by)
        df_ideal = df_ideal.sort_values(sort_by)
        if reindex:
            df_ideal.index = range(len(df_ideal))
            df.index = range(len(df))
        if set_dtypes:
            for column, dtype in df_ideal.dtypes.to_dict().items():
                df[column] = df[column].astype(dtype)
        assert df_ideal.equals(df), 'Итоговый датафрейм не совпадает с верным результатом.'


