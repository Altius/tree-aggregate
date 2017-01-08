/*
  FILE: dtree-types.hpp
  AUTHOR: Shane Neph
  CREATE DATE: Jan.2017
*/
//
//    Decision Tree
//    Copyright (C) 2016-2017 Altius Institute for Biomedical Sciences
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License along
//    with this program; if not, write to the Free Software Foundation, Inc.,
//    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#ifndef __ALTIUS_DTREE_TYPES_HPP__
#define __ALTIUS_DTREE_TYPES_HPP__

#include <map>
#include <string>

#include <boost/dynamic_bitset.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

namespace Tree {

  namespace ub = boost::numeric::ublas;

  std::string FileFormatVersion = "0.1";

  typedef std::size_t featureID;
  typedef std::size_t label;
  typedef float dtype;
  typedef boost::dynamic_bitset<> Monitor;

  template <typename T>
  using ColumnVector = ub::vector<T>;
  typedef ub::compressed_vector<dtype> RowVector;
  typedef ColumnVector<RowVector> DataVectorVector;
  typedef ub::generalized_vector_of_vector<dtype, ub::column_major, DataVectorVector> DataMatrix;
  typedef ub::generalized_vector_of_vector<dtype, ub::row_major, DataVectorVector> DataMatrixInv;

  typedef std::map<boost::dynamic_bitset<>, featureID> FeatureMap;
  FeatureMap featureMap;

  enum struct Criterion { GINI, ENTROPY };
  enum struct SplitStrategy { BEST, RANDOM };
  enum struct SplitMaxFeat { INT, FLT, SQRT, LOG2 };
  enum struct ClassWeightType { SAME, BALANCED };

} // namespace Tree

#endif // __ALTIUS_DTREE_TYPES_HPP__
