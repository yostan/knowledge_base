package com.gz.dt

import org.apache.spark.mllib.linalg.Vector

/**
 *  created by naonao on 15/6/12.
 */
class VectorUtils extends Serializable {

}

object VectorUtils {
  def dotProduct(vec1: Vector, vec2: Vector): Double = {
    val length1 = vec1.size
    val length2 = vec2.size

    if(length1 != length2) {
      throw new IllegalArgumentException("Vector of diff length")
    }

    var ans = 0.0
    var i = 0
    while (i < length1) {
      ans += vec1(i) * vec2(i)
      i += 1
    }
    ans
  }
}
