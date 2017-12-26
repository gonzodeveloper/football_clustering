# Kyle Hart
# 18 Sept 2017
#
# Project: football_predictions
# File: cluster_analysis.py
# Description:  Application of Spark's Gaussian Mixture tool for cluster analysis.
#           Building clusters around teams in the NCAAF football dataset


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import GaussianMixture
import MySQLdb
import os


if __name__ == "__main__":
    # Get Spark context
    print("Configuring Spark... \n")
    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
    sc = SparkContext("local[4]", "ratings filter")
    sc.setLogLevel(logLevel="OFF")
    spark = SparkSession(sparkContext=sc)

    #Connect to database
    print("Connecting to Database... \n")
    conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='Rafik!is@RapG0d', db='league')
    curs = conn.cursor()

    query = "SELECT " \
            "   team_id, " \
            "   team_name, " \
            "   rush_att, " \
            "   rush_net, " \
            "   pass_att, " \
            "   pass_net, " \
            "   ttl_td, " \
            "   intc_cmp, " \
            "   opp_score " \
            "FROM team_avgs"
    query = "SELECT " \
            "   team_id, " \
            "   team_name, " \
            "   AVG(t1_rush), " \
            "   AVG(t1_pass), " \
            "   AVG(t2_rush), " \
            "   AVG(t2_pass) " \
            "FROM full_game_stats " \
            "JOIN team ON 1=1 " \
            "   AND full_game_stats.t1_id = team.team_id " \
            "GROUP BY team_id, team_name"
    curs.execute(query)
    sql_dat = curs.fetchall()
    team_ids = [row[0] for row in sql_dat]
    team_names = [row[1] for row in sql_dat]
    features = [row[2:] for row in sql_dat]

    data = sc.parallelize(features, 1)
    model = GaussianMixture.train(data, k=10)
    cluster_labels = model.predict(data).collect()


    labels = zip(team_ids,team_names, cluster_labels)
    df = spark.createDataFrame( labels,
                        ["team_id", "team_name", "cluster_id"] )
    df.createOrReplaceTempView("model")
    for k in range(10):
        spark.sql("SELECT * FROM model WHERE cluster_id = {}".format(k)).show()
