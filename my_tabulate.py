import pandas
import tabulate


def my_tabulate(data, tablefmt='pipe', **params):
    if isinstance(data, pandas.DataFrame):
        df = data
        data = df.reset_index().values.tolist()
        if 'headers' not in params:
            params['headers'] = list(df.columns)
    assert isinstance(data, list), type(data)
    if data == [] and 'headers' in params:
        data = [(None for _ in params['headers'])]
    tabulate.MIN_PADDING = 0
    return tabulate.tabulate(data, tablefmt=tablefmt, **params)
