package com.gz.dt

import org.ansj.splitWord.analysis.ToAnalysis
import org.ansj.util.FilterModifWord
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.UnionRDD
import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.JavaConversions._

/**
 *  created by naonao on 15/6/11.
 */
class NB {

}

object NB {
  def main(args: Array[String]) {
    if(args.length < 3){
      System.err.println("Usage: NB <dir-01> <dir-02> <dir-03> <path-stopword>")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("N-B")//.setMaster("local[4]")
    val sc = new SparkContext(conf)

    val rdd1 = sc.wholeTextFiles(args(0),4)
    val rdd2 = sc.wholeTextFiles(args(1),4)
    //val rdd3 = sc.wholeTextFiles(args(2),4)

    val stopWord = sc.textFile(args(2)).collect().toList

    val rdd = new UnionRDD[(String, String)](sc,Seq(rdd1, rdd2))

    val wordFilter = rdd.map{tup =>
      val terms = ToAnalysis.parse(tup._2)
      FilterModifWord.insertStopWords(stopWord)
      val filterTerms = FilterModifWord.modifResult(terms)

      val words = for(i <- 0 until filterTerms.size()) yield filterTerms.get(i).getName
      words.mkString("\t")
    }.map(wordInDoc => wordInDoc.split("\t").toSeq)

    val docType = rdd.map(x => x._1.substring(x._1.lastIndexOf("/")-7, x._1.lastIndexOf("/")).hashCode.toLong)

    val hashTF = new HashingTF()
    val tf = hashTF.transform(wordFilter)

    val corpus = docType.zip(tf)

    val labeledPoint = corpus.map(lp => LabeledPoint(lp._1, lp._2))
    val splits = labeledPoint.randomSplit(Array(0.7, 0.3), seed = 11L)

    val train = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(train, lambda = 1.0)

    val predictAndReal = test.map(p => (model.predict(p.features), p.label))

    val accuracy = 1.0 * predictAndReal.filter(tup => tup._1 == tup._2).count() / test.count()

    println(s"---------------- accuracy(nb-tf): $accuracy")

    sc.stop()

  }
}
