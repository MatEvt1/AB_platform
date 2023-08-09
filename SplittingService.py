from pydantic import BaseModel


class Experiment(BaseModel):
    """
    id - идентификатор эксперимента.
    buckets_count - необходимое количество бакетов.
    conflicts - список идентификаторов экспериментов, которые нельзя проводить
        одновременно на одних и тех же пользователях.
    """
    id: int
    buckets_count: int
    conflicts: list[int] = []

class SplittingService:

    def __init__(self, buckets_count, bucket_salt, buckets=None, id2experiment=None):
        """Класс для распределения экспериментов и пользователей по бакетам.

        :param buckets_count (int): количество бакетов.
        :param bucket_salt (str): соль для разбиения пользователей по бакетам.
            При одной соли каждый пользователь должен всегда попадать в один и тот же бакет.
            Если изменить соль, то распределение людей по бакетам должно измениться.
        :param buckets (list[list[int]]) - список бакетов, в каждом бакете перечислены идентификаторы
            эксперименты, которые в нём проводятся.
        :param id2experiment (dict[int, Experiment]) - словарь пар: идентификатор эксперимента - эксперимент.
        """
        self.buckets_count = buckets_count
        self.bucket_salt = bucket_salt
        if buckets:
            self.buckets = buckets
        else:
            self.buckets = [[] for _ in range(buckets_count)]
        if id2experiment:
            self.id2experiment = id2experiment
        else:
            self.id2experiment = {}

    def _get_hash_modulo(self, value: str, modulo: int, salt: str):
        """Вычисляем остаток от деления: (hash(value) + salt) % modulo."""
        hash_value = int(hashlib.md5(str.encode(value + salt)).hexdigest(), 16)
        return hash_value % modulo

    def process_user(self, user_id):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить бакет пользователя.
        Затем для каждого эксперимента в этом бакете выбрать пилотную или контрольную группу.

        :param user_id (str): идентификатор пользователя
        :return bucket_id, experiment_groups:
            - bucket_id (int) - номер бакета (индекс элемента в self.buckets)
            - experiment_groups (list[tuple]) - список пар: id эксперимента, группа.
                Группы: 'A', 'B'.
            Пример: (8, [(194, 'A'), (73, 'B')])
        """
        bucket_id = self._get_hash_modulo(user_id, self.buckets_count, self.bucket_salt)
        experiment_ids = self.buckets[bucket_id]
        experiments = [
            self.id2experiment[experiment_id] for experiment_id in experiment_ids
        ]

        experiment_groups = []
        for experiment in experiments:
            second_hash = self._get_hash_modulo(user_id, 2, experiment.salt)
            group = 'B' if second_hash == 1 else 'A'
            experiment_groups.append((experiment.id, group))
        return bucket_id, experiment_groups

    def add_experiment(self, experiment):
        """Проверяет можно ли добавить эксперимент, добавляет если можно.

        :param experiment (Experiment): параметры эксперимента, который нужно запустить
        :return success, buckets:
            success (boolean) - можно ли добавить эксперимент, True - можно, иначе - False
            buckets (list[list[int]]]) - список бакетов, в каждом бакете перечислены идентификаторы эксперименты,
                которые в нём проводятся.
        """
        # список из элементов [bucket_id, количество совместных экспериментов]
        available_buckets_meta = []
        for bucket_id, bucket in enumerate(self.buckets):
            if set(experiment.conflicts) & set(bucket):
                continue
            available_buckets_meta.append((bucket_id, len(bucket)))
        if len(available_buckets_meta) < experiment.buckets_count:
            return False, self.buckets
        sorted_available_buckets_meta = sorted(available_buckets_meta, key=lambda x: -x[1])
        for bucket_id, _ in sorted_available_buckets_meta[:experiment.buckets_count]:
            self.buckets[bucket_id].append(experiment.id)
        return True, self.buckets
