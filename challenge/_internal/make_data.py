from challenge import DATA_DIR
import pandas as pd
import numpy as np
import logging

evaluating_years = [2013, 2014, 2015]

fmt = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(level="INFO", format=fmt)
logger = logging.getLogger("make_data")

if __name__ == '__main__':
    logger.info("Loading HNP_StatsData.csv")
    raw = pd.read_csv(DATA_DIR / 'HNP_StatsData.csv')

    # 2018 might be only partial results.
    raw.drop(columns=['2018'], inplace=True)

    # create tidy dataframe in long format.
    logger.info("Creating tidy dataframe.")
    columns_years = raw.columns[raw.columns.str.isnumeric()]
    tidy = (raw
            .melt(id_vars=['Country Code', 'Indicator Code'], 
                  value_vars=columns_years, 
                  var_name='year')
            .rename(columns={'Country Code': 'country', 
                             'Indicator Code': 'indicator'})
            .astype({'country': 'category', 
                     'indicator': 'category', 
                     'year': int, 
                     'value': float}))

    # In [1]: tidy.sample(5)
    # Out[1]:
    #         country          indicator  year         value
    # 3341198     CEB  SH.STA.OWAD.FE.ZS  1992  4.579314e+01
    # 5964244     TSA  SP.POP.1564.MA.IN  2017  6.067836e+08
    # 2244313     ISL     SP.DYN.SMAM.MA  1981  2.610000e+01
    # 227091      ASM  SH.XPD.OOPC.PP.CD  1962           NaN
    # 1185926     CYP     SP.POP.7579.FE  1971  6.842000e+03

    # guarantee at least one point for testing
    logger.info("Filtering tidy dataframe.")
    candidates = (tidy
                  .query("year.isin(@evaluating_years)")
                  .dropna()
                  [lambda df: df['value'].notnull()]
                  .set_index(['indicator', 'country'])
                  .index
                  .values)

    # guarantee good training data
    good = (tidy
            .assign(value=tidy.value.notnull())
            .query("year >= 2000")
            .groupby(['indicator', 'country'])
            ['value']
            .mean()
            .pipe(lambda s: s[s > 0.7])
            .index
            .values)

    mask = np.intersect1d(good, candidates)
    tidy_filtered = (tidy
                     .set_index(['indicator', 'country'])
                     .loc[mask]
                     .dropna()
                     .reset_index())

    # create training and testing dataframes
    logger.info("Creating answers.csv")
    answers = (tidy_filtered
               .query("year.isin(@evaluating_years)")
               .reset_index(drop=True))

    logger.info("Creating data.csv")
    data = (tidy_filtered
            .query("~year.isin(@evaluating_years)")
            .reset_index(drop=True))

    logger.info("Creating test.csv")
    test = answers.drop(columns=['value'])


    # Sanity checks
    logger.info("Running sanity checks")
    assert tidy_filtered['year'].nunique() == 58
    assert tidy_filtered['country'].nunique() == 258
    assert tidy_filtered['indicator'].nunique() == 321
    assert len(mask) == 56163
    assert not tidy_filtered.duplicated(subset=['indicator', 'country', 'year']).any()
    assert not (pd.concat([data[['indicator', 'country', 'year']], 
                           test[['indicator', 'country', 'year']]])
                  .duplicated()
                  .any())
    assert len(data) > len(test)

    # save data
    logger.info(f"Saving {DATA_DIR}/{{data,test,answers}}.csv")
    test.to_csv(DATA_DIR / 'test.csv', index=False)
    data.to_csv(DATA_DIR / 'data.csv', index=False)
    answers.to_csv(DATA_DIR / 'answers.csv', index=False)
