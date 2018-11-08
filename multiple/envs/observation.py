import pandas as pd


"""""

Read observation from thermal manikin
"""""


class thermalManikin():
    """
    Data from excel log of thermal manikin

    Parameters
    ----------
    excelPath: str,
    """

    def __init__(self, excelPath):  
        df = pd.read_csv(excelPath, header=3, encoding = "ISO-8859-1")
        df.index = pd.to_datetime(df['Unnamed: 0'])
        cols = [c for c in df.columns if 'Unname' not in c]
        df = df[cols]
        #df = df.drop(df.index[1])
 
        print("Column headings:")
        print(df.last('60s'))



db = thermalManikin("Maniki.csv")


