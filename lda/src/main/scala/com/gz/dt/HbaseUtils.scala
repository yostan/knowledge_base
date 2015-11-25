package com.gz.dt

import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapred.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{HColumnDescriptor, HTableDescriptor, HBaseConfiguration}
import org.apache.hadoop.hbase.client.{Put, HBaseAdmin}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.rdd.{PairRDDFunctions, RDD}

/**
 * Created by naonao on 2015/6/12
 */
class HbaseUtils(val tableName: String, val zkQuorum: String, val colFamily: String) extends Serializable {

  def createTableAndWrite(rdd: RDD[(String, String)]): Unit ={
    val conf = HBaseConfiguration.create()
    conf.set("hbase.zookeeper.quorum", zkQuorum)
    conf.set("hbase.zookeeper.property.clientPort", "2181")

    val admin = new HBaseAdmin(conf)
    if(!admin.isTableAvailable(tableName)){
      val tableDesc = new HTableDescriptor(tableName)
      tableDesc.addFamily(new HColumnDescriptor(colFamily))
      admin.createTable(tableDesc)
    }

    //new api
    //val job = Job.getInstance(conf)
    //job.getConfiguration.set(TableOutputFormat.OUTPUT_TABLE,tableName)
    //job.setOutputFormatClass(classOf[TableOutputFormat[Put]])
    //job.setOutputFormatClass(classOf[NullOutputFormat[ImmutableBytesWritable,Put]])

    //old api
    val job = new JobConf(conf)
    job.set(TableOutputFormat.OUTPUT_TABLE, tableName)
    job.setOutputFormat(classOf[TableOutputFormat])

    new PairRDDFunctions[ImmutableBytesWritable, Put](rdd.map{case (k, v) => putData(k, v)}).saveAsHadoopDataset(job)

  }

  def putData(srcDoc: String, simDocs: String) ={
    val record = new Put(Bytes.toBytes(srcDoc))
    record.add(Bytes.toBytes(colFamily), Bytes.toBytes("simdoc"), Bytes.toBytes(simDocs))
    (new ImmutableBytesWritable(), record)
  }
}
