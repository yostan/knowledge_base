package com.gz.dt

import org.ansj.splitWord.analysis.ToAnalysis
import org.ansj.util.FilterModifWord
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.feature.{Normalizer, HashingTF}
import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.JavaConversions._
/**
 * Created by naonao on 2015/6/12
 */
class LDASimilar {

}

object LDASimilar {
  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: LDASimilar <path-docs> <path-stopword> <hbase-tablename> <hbase-zkquorum> <hbase-colfamily>")
    }

    val conf = new SparkConf().setAppName("lda-similar")
    val sc = new SparkContext(conf)

    val rdd1 = sc.wholeTextFiles(args(0),4)
    val stopWord = sc.textFile(args(1),2).collect().toList

    val filterWord = rdd1.map{tup =>
      val terms = ToAnalysis.parse(tup._2)
      FilterModifWord.insertStopWords(stopWord)
      val filterTerms = FilterModifWord.modifResult(terms)

      val words = for(i <- 0 until filterTerms.size()) yield filterTerms.get(i).getName
      words.mkString("\t")
    }.map(wordInDoc => wordInDoc.split("\t").toSeq)

    val docId = rdd1.map(x => x._1.substring(x._1.lastIndexOf("/"), x._1.lastIndexOf(".")).toLong)

    val hashTF = new HashingTF()
    val tf = hashTF.transform(filterWord)

    val corpus = docId.zip(tf)

    val ldaModel = new LDA().setK(4).run(corpus)

    val doc_topic = ldaModel.topicDistributions

    val norm2 = new Normalizer()
    val doc_topic_norm2 = doc_topic.map(dt => (dt._1, norm2.transform(dt._2)))

    //computer top5 sim-doc
    val docId_array = docId.collect()
    for (i <- docId_array) {
      val start_vec = doc_topic_norm2.filter(x => x._1 == i).map(v => v._2).take(1)(0)
      val tup3 = doc_topic_norm2.map{v =>
        val sim = VectorUtils.dotProduct(start_vec, v._2)
        (i.toString, v._1.toString, sim)
      }
      val top5 = tup3.sortBy(x => x._3, false).take(6).filter(t => t._1 !=t._2).map(y => (y._1, y._2))

      val sb = new StringBuilder()
      val top5_rdd = sc.parallelize(top5).groupByKey()
      val top5_value = top5_rdd.mapValues{iter =>
        val it = iter.iterator
        while(it.hasNext){
          sb.append(it.next()+"|")
        }
        sb.substring(0, sb.length-1)
      }

      val hbaseOps = new HbaseUtils(args(2), args(3), args(4))
      hbaseOps.createTableAndWrite(top5_value)

    }

    sc.stop()
  }
}
