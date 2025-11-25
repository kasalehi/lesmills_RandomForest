import pandas as pd
from src.logger import logger
class transforming:
    
    def __init__(self, dataset:pd.DataFrame=''):
        self.dataset=dataset
    def read_data(self):
        # lets read the data from sql database
        from sqlalchemy import create_engine
        import pandas as pd
        from urllib.parse import quote_plus

        server   = r'LMNZLREPORT01\LM_RPT'                 # named instance
        database = 'LMNZ_Report'
        driver   = 'ODBC Driver 17 for SQL Server'         # use 18 if installed

        username = 'LMNZ_ReportUser'
        password = 'LMNZ_ReportUser'                           # avoid @ : / ? in raw form

        # If password contains special chars, do: password = quote_plus(password)
        engine = create_engine(
            f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}/{database}"
            f"?driver={driver.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=yes"
        )

        query = """

        WITH base AS (
            SELECT
                cm.MembershipID,
                cm.Origin,
                cm.[Status Desc],
                cm.SubCategory,
                cm.Term,
                cm.PaymentFrequency,
                cm.RegularPayment,
                cm.Gender,
                -- Age: no need to ROUND, DATEDIFF already returns INT
                DATEDIFF(YEAR, cm.DOB, GETDATE()) AS Age,
                -- Churn flag
                CASE 
                    WHEN cm.[Status Desc] = 'Inactive' THEN 1
                    WHEN cm.[Status Desc] = 'Active'   THEN 0
                    ELSE NULL
                END AS Churned,
                att.WeekVisits
            FROM fact.LMNZ_ALLMemberships AS cm
            JOIN repo.MemberWeeklyAttendanceCounts AS att
                ON cm.MembershipID = att.MembershipID
            WHERE cm.Term IN ('12 Months', '12 Month', '24 Month', '24 Months')
            AND cm.[Status Desc] IN ('Active', 'Inactive')   -- keep only relevant statuses
        ),
        total as (
        SELECT
            MembershipID,
            Origin,
            SubCategory,
            Term,
            PaymentFrequency,
            RegularPayment,
            Gender,
            Age,
            SUM(WeekVisits) AS TotalAttendance,
            Churned
        FROM base
        GROUP BY
            MembershipID,
            Origin,
            [Status Desc],
            SubCategory,
            Term,
            PaymentFrequency,
            RegularPayment,
            Gender,
            Age,
            Churned)
            select * from total;

        """
        df = pd.read_sql_query(query, engine)
        logger.info("read data from sql database")
        return df
    
    def transform_data(self, df:pd.DataFrame)->pd.DataFrame:
        logger.info("starting transformaing..")
        churned=df[df['Churned']==1]
        Not_Churned=df[df['Churned']==0]
        logger.info("fixing abnormality using sample data...")
        Churned_sampled=churned.sample(n=Not_Churned.shape[0],random_state=4)
        final_df=pd.concat([Churned_sampled,Not_Churned],axis=0)
        final_df=final_df.sample(frac=1,random_state=42).reset_index(drop=True)
        final_df=final_df.drop('MembershipID',axis=1)
        logger.info("datasource created and prepare for training")
        return final_df
    def saving(self, data: pd.DataFrame):
        # Base folder where you want to save the CSV
        from pathlib import Path
        base_dir = Path(
            r"C:\Users\ksalehi\OneDrive - Les Mills New Zealand Limited\Desktop\lesmills\lesmills_RandomForest\datasets"
        )

        logger.info("saving data in datasets folder")

        # Create the folder if it doesn't exist
        base_dir.mkdir(parents=True, exist_ok=True)

        # Build the full path to the CSV
        out_path = base_dir / "transform_data.csv"

        # Save CSV (no extra spaces, no missing folders)
        data.to_csv(out_path, index=False)

        logger.info(f"data saved to {out_path}")


# lets defin objects
if __name__=="__main__":
    obj=transforming()
    reading=obj.read_data()
    trans=obj.transform_data(reading)
    save=obj.saving(trans)