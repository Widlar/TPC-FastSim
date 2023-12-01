import os
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_PATH = Path(os.path.realpath(__file__)).parent
_VERSION = 'data_v1'
_V4PLUS_VAR_NAMES = ["crossing_angle", "dip_angle", "drift_length", "pad_coordinate", "row", "pT"]
_V4PLUS_VAR_TYPES = [float, float, float, float, int, float]

def multiple_lines_evtId(data_df):
    # Присваиваем новый evtId путем целочисленного деления текущего evtId на 3
    data_df['new_evtId'] = data_df['evtId'] // 3

    # Удаляем старый столбец 'evtId'
    data_df.drop('evtId', axis=1, inplace=True)

    # Получаем список столбцов без 'evtId'
    columns = data_df.columns.tolist()

    # Удаляем 'new_evtId' из списка, чтобы избежать дублирования при переупорядочивании
    columns.remove('new_evtId')

    # Вставляем 'new_evtId' на первое место в список столбцов
    columns.insert(0, 'new_evtId')

    # Переупорядочиваем DataFrame с новым порядком столбцов
    data_df = data_df.reindex(columns=columns)

    # Переименовываем 'new_evtId' в 'evtId'
    data_df.rename(columns={'new_evtId': 'evtId'}, inplace=True)

    # Сортировка DataFrame по новому 'evtId'
    data_df.sort_values(['evtId', 'row'], inplace=True)
    return data_df


class Reader:
    def __init__(self, variables, types):
        assert len(variables) == len(types), 'Reader.__init__: variables and types have different length'
        self.vars = variables
        self.types = types
        self.data = []


    def read_line(self, line, index):
        stems = line.split()
        assert (len(stems) // len(self.vars)) * len(self.vars) == len(stems), [line, self.vars]

        for i_group in range(0, len(stems), len(self.vars)):
            self.data.append((index,) + tuple(_T(stems[i_group + i_var]) for i_var, _T in enumerate(self.types)))

    def build(self):
        data_df= pd.DataFrame(self.data, columns=['evtId'] + self.vars).set_index('evtId')
        return data_df


def raw_to_csv(fname_in=None, fname_out=None, concat_ids=False) :
    if fname_in is None:
        fname_in = str(_THIS_PATH.joinpath(_VERSION, 'raw', 'digits.dat'))
    if fname_out is None:
        csv_path = _THIS_PATH.joinpath(_VERSION, 'csv')
        if not os.path.isdir(csv_path):
            csv_path.mkdir()
        fname_out = str(csv_path.joinpath('digits.csv'))

    with open(fname_in, 'r') as f:
        lines = f.readlines()

    reader_main = Reader(variables=['ipad', 'itime', 'amp'], types=[int, int, float])

    data_sources = [lines]
    readers = [reader_main]

    if 'params:' in lines[0]:
        assert len(lines) % 2 == 0, 'raw_to_csv: Odd number of lines when expected even'

        if _VERSION == 'data_v2':
            reader_features = Reader(variables=["crossing_angle", "dip_angle"], types=[float, float])
        elif _VERSION == 'data_v3':
            reader_features = Reader(
                variables=["crossing_angle", "dip_angle", "drift_length"], types=[float, float, float]
            )
        elif _VERSION == 'data_v4':
            reader_features = Reader(
                variables=["crossing_angle", "dip_angle", "drift_length", "pad_coordinate"],
                types=[float, float, float, float],
            )
        elif (_VERSION == 'data_v4plus') or (_VERSION == 'data_v4plus_shortFixedPt'):
            reader_features = Reader(
                variables=_V4PLUS_VAR_NAMES,
                types=_V4PLUS_VAR_TYPES,
                concat_ids = True
            )
        elif (_VERSION == 'data_v5'):
            reader_features = Reader(
                variables=_V4PLUS_VAR_NAMES,
                types=_V4PLUS_VAR_TYPES,
            )
        else:
            raise NotImplementedError

        lines, lines_angles = lines[1::2], lines[::2]
        lines_angles = [' '.join(line.split()[1:]) for line in lines_angles]

        data_sources = [lines, lines_angles]
        readers = [reader_main, reader_features]

    for evt_id, lines_tuple in enumerate(zip(*data_sources)):
        for r, l in zip(readers, lines_tuple):
            r.read_line(l, evt_id)

    result = pd.concat([r.build() for r in readers], axis=1).reset_index()
    if concat_ids:
        result =  data_df = multiple_lines_evtId(result)
    result.to_csv(fname_out, index=False)


def read_csv_2d(filename=None, pad_range=(40, 50), time_range=(265, 280), strict=True, misc_out=None):
    if filename is None:
        filename = str(_THIS_PATH.joinpath(_VERSION, 'csv', 'digits.csv'))

    df = pd.read_csv(filename)

    def sel(df, col, limits):
        return (df[col] >= limits[0]) & (df[col] < limits[1])

    if 'drift_length' in df.columns:
        df['itime'] -= df['drift_length'].astype(int)

    if 'pad_coordinate' in df.columns:
        df['ipad'] -= df['pad_coordinate'].astype(int)

    selection = sel(df, 'itime', time_range) & sel(df, 'ipad', pad_range)

    def convert_event(event):
        result = np.zeros(dtype=float, shape=(pad_range[1] - pad_range[0], time_range[1] - time_range[0]))
        indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
        result[indices] = event.amp.values
        return result

    def convert_event_group(event_group):
        group_data = []
        for _, event in event_group.groupby('evtId'):
            result = np.zeros(dtype=float, shape=(pad_range[1] - pad_range[0], time_range[1] - time_range[0]))
            indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
            result[indices] = event.amp.values
            group_data.append(result)
        return np.stack(group_data, axis=-1)

    if _VERSION == 'data_v5':
        df['group_id'] = df['evtId'] // 3
        g = df[selection].groupby('group_id')
        bad_ids = df[~selection]['group_id'].unique()
        anti_selection = df['group_id'].apply(lambda x: x in bad_ids)
        anti_g = df[anti_selection].groupby('group_id')
        data = np.stack(g.apply(convert_event_group).values)
        # data = np.moveaxis(data, -1, 0)  # Изменение формы с (20000, 8, 16, 3) на (3, 20000, 8, 16)

    else:
        g = df[selection].groupby('evtId')
        bad_ids = df[~selection]['evtId'].unique()
        anti_selection = df['evtId'].apply(lambda x: x in bad_ids)
        anti_g = df[anti_selection].groupby('evtId')
        data = np.stack(g.apply(convert_event).values)

    if not selection.all():
        msg = (
            f"WARNING: current selection ignores {(~selection).sum() / len(selection) * 100}% of the data"
            f" ({len(anti_g)} events)!"
        )
        assert not strict, msg
        print(msg)

    features = None
    anti_data = None
    if not selection.all() and misc_out is not None:
        assert isinstance(misc_out, dict)
        pad_range = [df[anti_selection]["ipad"].min(), df[anti_selection]["ipad"].max() + 1]
        time_range = [df[anti_selection]["itime"].min(), df[anti_selection]["itime"].max() + 1]
        anti_data = np.stack(anti_g.apply(convert_event).values)
        misc_out["anti_data"] = anti_data
        misc_out["bad_ids"] = bad_ids

    if 'crossing_angle' in df.columns:
        features = ['crossing_angle', 'dip_angle']
        if 'drift_length' in df.columns:
            features += ['drift_length']
        if 'pad_coordinate' in df.columns:
            features += ['pad_coordinate']
        if "row" in df.columns:
            features += ["row"]
        if "pT" in df.columns:
            features += ["pT"]

        feature_cols = []
        if 'crossing_angle' in df.columns:
            feature_cols.append('crossing_angle')
        if 'dip_angle' in df.columns:
            feature_cols.append('dip_angle')
        if 'drift_length' in df.columns:
            feature_cols.append('drift_length')
        if 'pad_coordinate' in df.columns:
            feature_cols.append('pad_coordinate')
        if 'row' in df.columns:
            feature_cols.append('row')
        if 'pT' in df.columns:
            feature_cols.append('pT')

        if _VERSION != 'data_v5':
            assert (
                (g[features].std() == 0).all(axis=1) | (g[features].size() == 1)
            ).all(), 'Varying features within same events...'

        if _VERSION == 'data_v5':
            # Определение списка признаков (feature columns)

            group_features = []
            for _, group in g:
                features_mean = group[feature_cols].mean().values
                group_features.append(np.stack([features_mean] * 3, axis=-1))
            features = np.stack(group_features)
            # features = np.moveaxis(features,  -1, 0)  #


        else:
            features = g[feature_cols].mean().values

    return data, features
