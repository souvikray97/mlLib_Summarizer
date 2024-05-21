

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]): Unit = {

    // Create a Spark session
    val spark = SparkSession.builder
      .appName("Spark ML Summarizer")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Sample data
    val data = Seq(
      (Vectors.dense(2.0, 3.0, 5.0), 1.0),
      (Vectors.dense(4.0, 6.0, 7.0), 2.0)
    )

    // Convert the sequence to a DataFrame
    val df = data.toDF("features", "weight")

    // Calculate mean and variance with weight
    val (meanVal, varianceVal) = df.select(Summarizer.metrics("mean", "variance")
        .summary($"features", $"weight").as("summary"))
      .select("summary.mean", "summary.variance")
      .as[(Vector, Vector)].first()

    println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

    // Calculate mean and variance without weight
    val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
      .as[(Vector, Vector)].first()

    println(s"without weight: mean = ${meanVal2}, variance = ${varianceVal2}")

    // Stop the Spark session
    spark.stop()
  }
}
