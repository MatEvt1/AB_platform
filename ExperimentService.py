import numpy as np
from pydantic import BaseModel
from scipy import stats
import pandas as pd


class Design(BaseModel):
    """Дата-класс с описание параметров эксперимента.

    statistical_test - тип статтеста. ['ttest', 'bootstrap']
    effect - размер эффекта в процентах
    alpha - уровень значимости
    beta - допустимая вероятность ошибки II рода
    bootstrap_iter - количество итераций бутстрепа
    bootstrap_ci_type - способ построения доверительного интервала. ['normal', 'percentile', 'pivotal']
    bootstrap_agg_func - метрика эксперимента. ['mean', 'quantile 95']
    """
    statistical_test: str
    effect: float
    alpha: float = 0.05
    beta: float = 0.1
    bootstrap_iter: int = 1000
    bootstrap_ci_type: str
    bootstrap_agg_func: str

class ExperimentsService:

    @staticmethod
    def _calc_strat_mean(df: pd.DataFrame, weights: pd.Series) -> float:
        """Считает стратифицированное среднее.

        :param df: датафрейм с целевой метрикой и данными для стратификации
        :param weights: маппинг {название страты: вес страты в популяции}
        """
        strat_mean = df.groupby('strat')['metric'].mean()
        return (strat_mean * weights).sum()

    @staticmethod
    def _calc_strat_var(df: pd.DataFrame, weights: pd.Series) -> float:
        """Считает стратифицированную дисперсию.

        :param df: датафрейм с целевой метрикой и данными для стратификации
        :param weights: маппинг {название страты: вес страты в популяции}
        """
        strat_var = df.groupby('strat')['metric'].var()
        return (strat_var * weights).sum()

    def _ttest_strat(self, metrics_strat_a_group, metrics_strat_b_group):
        """Применяет постстратификацию, возвращает pvalue.

        Веса страт считаем по данным обеих групп.
        Предполагаем, что эксперимент проводится на всей популяции.
        Веса страт нужно считать по данным всей популяции.

        :param metrics_strat_a_group (np.ndarray): значения метрик и страт группы A.
            shape = (n, 2), первый столбец - метрики, второй столбец - страты.
        :param metrics_strat_b_group (np.ndarray): значения метрик и страт группы B.
            shape = (n, 2), первый столбец - метрики, второй столбец - страты.
        :param design (Design): объект с данными, описывающий параметры эксперимента
        :return (float): значение p-value
        """
        weights = (
            pd.Series(np.hstack((metrics_strat_a_group[:, 1], metrics_strat_b_group[:, 1],)))
            .value_counts(normalize=True)
        )
        a = pd.DataFrame(metrics_strat_a_group, columns=['metric', 'strat'])
        b = pd.DataFrame(metrics_strat_b_group, columns=['metric', 'strat'])
        a_strat_mean = self._calc_strat_mean(a, weights)
        b_strat_mean = self._calc_strat_mean(b, weights)
        a_strat_var = self._calc_strat_var(a, weights)
        b_strat_var = self._calc_strat_var(b, weights)
        delta = b_strat_mean - a_strat_mean
        std = (a_strat_var / len(a) + b_strat_var / len(b)) ** 0.5
        t = delta / std
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(t)))
        return pvalue

    def get_pvalue(self, metrics_strat_a_group, metrics_strat_b_group, design):
        """Применяет статтест, возвращает pvalue.

        :param metrics_strat_a_group (np.ndarray): значения метрик и страт группы A.
            shape = (n, 2), первый столбец - метрики, второй столбец - страты.
        :param metrics_strat_b_group (np.ndarray): значения метрик и страт группы B.
            shape = (n, 2), первый столбец - метрики, второй столбец - страты.
        :param design (Design): объект с данными, описывающий параметры эксперимента
        :return (float): значение p-value
        """
        if design.statistical_test == 'ttest':
            if design.stratification == 'off':
                _, pvalue = stats.ttest_ind(metrics_strat_a_group[:, 0], metrics_strat_b_group[:, 0])
                return pvalue
            elif design.stratification == 'on':
                return self._ttest_strat(metrics_strat_a_group, metrics_strat_b_group)
            else:
                raise ValueError('Неверный design.stratification')
        else:
            raise ValueError('Неверный design.statistical_test')

    def _create_group_generator(self, metrics, sample_size, n_iter):
        """Генератор случайных групп.

        :param metrics (pd.DataFame): таблица с метриками, columns=['user_id', 'metric'].
        :param sample_size (int): размер групп (количество пользователей в группе).
        :param n_iter (int): количество итераций генерирования случайных групп.
        :return (np.array, np.array): два массива со значениями метрик в группах.
        """
        user_ids = metrics['user_id'].unique()
        for _ in range(n_iter):
            a_user_ids, b_user_ids = np.random.choice(user_ids, (2, sample_size), False)
            a_metric_values = metrics.loc[metrics['user_id'].isin(a_user_ids), 'metric'].values
            b_metric_values = metrics.loc[metrics['user_id'].isin(b_user_ids), 'metric'].values
            yield a_metric_values, b_metric_values

    def _estimate_errors(self, group_generator, design, effect_add_type):
        """Оцениваем вероятности ошибок I и II рода.

        :param group_generator: генератор значений метрик для двух групп.
        :param design (Design): объект с данными, описывающий параметры эксперимента.
        :param effect_add_type (str): способ добавления эффекта для группы B.
            - 'all_const' - увеличить всем значениям в группе B на константу (b_metric_values.mean() * effect / 100).
            - 'all_percent' - увеличить всем значениям в группе B в (1 + effect / 100) раз.
        :return pvalues_aa (list[float]), pvalues_ab (list[float]), first_type_error (float), second_type_error (float):
            - pvalues_aa, pvalues_ab - списки со значениями pvalue
            - first_type_error, second_type_error - оценки вероятностей ошибок I и II рода.
        """
        pvalues_aa = []
        pvalues_ab = []

        for a_metric_values, b_metric_values in group_generator:
            pvalues_aa.append(self.get_pvalue(a_metric_values, b_metric_values, design))
            b_metric_values_with_effect = b_metric_values.copy()
            if effect_add_type == 'all_percent':
                b_metric_values_with_effect *= 1 + design.effect / 100
            elif effect_add_type == 'all_const':
                b_metric_values_with_effect += b_metric_values_with_effect.mean() * design.effect / 100
            pvalues_ab.append(self.get_pvalue(a_metric_values, b_metric_values_with_effect, design))
        first_type_error = np.mean(np.array(pvalues_aa) < design.alpha)
        second_type_error = np.mean(np.array(pvalues_ab) >= design.alpha)
        return pvalues_aa, pvalues_ab, first_type_error, second_type_error

    def estimate_errors(self, metrics, design, effect_add_type, n_iter):
        """Оцениваем вероятности ошибок I и II рода.

        :param metrics (pd.DataFame): таблица с метриками, columns=['user_id', 'metric'].
        :param design (Design): объект с данными, описывающий параметры эксперимента.
        :param effect_add_type (str): способ добавления эффекта для группы B.
            - 'all_const' - увеличить всем значениям в группе B на константу (b_metric_values.mean() * effect / 100).
            - 'all_percent' - увеличить всем значениям в группе B в (1 + effect / 100) раз.
        :param n_iter (int): количество итераций генерирования случайных групп.
        :return pvalues_aa (list[float]), pvalues_ab (list[float]), first_type_error (float), second_type_error (float):
            - pvalues_aa, pvalues_ab - списки со значениями pvalue
            - first_type_error, second_type_error - оценки вероятностей ошибок I и II рода.
        """
        group_generator = self._create_group_generator(metrics, design.sample_size, n_iter)
        return self._estimate_errors(group_generator, design, effect_add_type)

    def estimate_sample_size(self, metrics, design):
        """Оцениваем необходимый размер выборки для проверки гипотезы о равенстве средних.

        Для метрик, у которых для одного пользователя одно значение просто вычислите размер групп по формуле.
        Для метрик, у которых для одного пользователя несколько значений (например, response_time),
            вычислите необходимый объём данных и разделите его на среднее количество значений на одного пользователя.
            Пример, если в таблице metrics 1000 наблюдений и 100 уникальных пользователей, и для эксперимента нужно
            302 наблюдения, то размер групп будет 31, тк в среднем на одного пользователя 10 наблюдений, то получится
            порядка 310 наблюдений в группе.

        :param metrics (pd.DataFrame): датафрейм со значениями метрик из MetricsService.
            columns=['user_id', 'metric']
        :param design (Design): объект с данными, описывающий параметры эксперимента
        :return (int): минимально необходимый размер групп (количество пользователей)
        """
        effect = design.effect
        alpha = design.alpha
        beta = design.beta
        std = np.std(metrics['metric'].values)
        mean = np.mean(metrics['metric'].values)
        epsilon = effect / 100 * mean
        coef = metrics['user_id'].nunique() / len(metrics)

        t_alpha = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
        t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)
        z_scores_sum_squared = (t_alpha + t_beta) ** 2
        sample_size = int(
            np.ceil(
                z_scores_sum_squared * (2 * std ** 2) / (epsilon ** 2) * coef
            )
        )
        return sample_size

    def _generate_bootstrap_metrics(self, data_one, data_two, design):
        """Генерирует значения метрики, полученные с помощью бутстрепа.

        :param data_one, data_two (np.array): значения метрик в группах.
        :param design (Design): объект с данными, описывающий параметры эксперимента
        :return bootstrap_metrics, pe_metric:
            bootstrap_metrics (np.array) - значения статистики теста псчитанное по бутстрепным подвыборкам
            pe_metric (float) - значение статистики теста посчитанное по исходным данным
        """
        bootstrap_data_one = np.random.choice(data_one, (len(data_one), design.bootstrap_iter))
        bootstrap_data_two = np.random.choice(data_two, (len(data_two), design.bootstrap_iter))
        if design.bootstrap_agg_func == 'mean':
            bootstrap_metrics = bootstrap_data_two.mean(axis=0) - bootstrap_data_one.mean(axis=0)
            pe_metric = data_two.mean() - data_one.mean()
            return bootstrap_metrics, pe_metric
        elif design.bootstrap_agg_func == 'quantile 95':
            bootstrap_metrics = (
                    np.quantile(bootstrap_data_two, 0.95, axis=0)
                    - np.quantile(bootstrap_data_one, 0.95, axis=0)
            )
            pe_metric = np.quantile(data_two, 0.95) - np.quantile(data_one, 0.95)
            return bootstrap_metrics, pe_metric
        else:
            raise ValueError('Неверное значение design.bootstrap_agg_func')

    def _run_bootstrap(self, bootstrap_metrics, pe_metric, design):
        """Строит доверительный интервал и проверяет значимость отличий с помощью бутстрепа.

        :param bootstrap_metrics (np.array): статистика теста, посчитанная на бутстрепных выборках.
        :param pe_metric (float): значение статистики теста посчитанное по исходным данным.
        :return ci, pvalue:
            ci [float, float] - границы доверительного интервала
            pvalue (float) - 0 если есть статистически значимые отличия, иначе 1.
                Настоящее pvalue для произвольного способа построения доверительного интервала с помощью
                бутстрепа вычислить не тривиально. Поэтому мы будем использовать краевые значения 0 и 1.
        """
        alpha = design.alpha
        if design.bootstrap_ci_type == 'normal':
            c = stats.norm.ppf(1 - alpha / 2)
            se = np.std(bootstrap_metrics)
            left, right = pe_metric - c * se, pe_metric + c * se
        elif design.bootstrap_ci_type == 'percentile':
            left, right = np.quantile(bootstrap_metrics, [alpha / 2, 1 - alpha / 2])
        elif design.bootstrap_ci_type == 'pivotal':
            right, left = 2 * pe_metric - np.quantile(bootstrap_metrics, [alpha / 2, 1 - alpha / 2])
        else:
            raise ValueError('Неверно значение design.bootstrap_ci_type')
        pvalue = float(left < 0 < right)
        return (left, right,), pvalue