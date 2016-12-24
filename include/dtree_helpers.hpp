/*
  FILE: dtree-input.hpp
  AUTHOR: Shane Neph
  CREATE DATE: Dec.2016
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

#ifndef __DECISION_TREE_INPUT__
#define __DECISION_TREE_INPUT__

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "dtree.hpp"
#include "utils.hpp"

namespace Tree {

inline featureID
get_features(const std::string& s, char d1, char d2, std::unordered_map<featureID, dtype>& mp) {
  static const std::size_t LabelID = 0;
  std::string::size_type pos1 = 0, pos2 = 0, posA = 0;
  std::string next;
  int maxFeatureLabel = 0;
  bool label = false;
  do {
    pos2 = s.find(d1, pos1);
    next = s.substr(pos1, pos2-pos1);
    posA = next.find(d2);
    if ( posA == std::string::npos ) {
      if ( label )
        throw std::domain_error("Feature does not have a value? " + next);
      label = true;
      mp.insert(std::make_pair(LabelID, Utils::convert<featureID>(next, Utils::Nums::PLUSINT)));
      pos1 = pos2+1;
      continue;
    }

    featureID b = static_cast<featureID>(std::atoi(next.substr(0, posA).c_str()));
    dtype c = static_cast<dtype>(std::atof(next.substr(posA+1).c_str()));
    if ( b > maxFeatureLabel )
      maxFeatureLabel = b;
    else if ( b < 0 )
      throw std::domain_error("Feature labels must be +int " + next);

    static const dtype zero = dtype(0);
    if ( c != zero )
      mp.insert(std::make_pair(b, static_cast<dtype>(c)));
    pos1 = pos2 + 1;
  } while ( pos2 != std::string::npos ); // do-while
  if ( 0 == maxFeatureLabel )
    throw std::domain_error("No features found in at least 1 row");
  return maxFeatureLabel + label;
}

std::tuple<DataMatrix*, DataMatrixInv*, featureID>
read_data(const std::string& source, bool oob, double oobPercent) {
  std::ifstream infile(source.c_str());
  if ( !infile )
    throw std::domain_error("Unable to find/open: " + source);

  // kind of crazy, but going to read input twice since I don't want to 'resize'
  //  a DataMatrix each time I add a row.
  Utils::ByLine bl;
  std::size_t nRows=0, nCols=0, nNonZeroes = 0, maxFeature = 0;
  std::unordered_map<featureID, dtype> featureList;
  while ( infile >> bl ) {
    if ( bl.empty() )
      throw std::domain_error("empty line found in input file");
    ++nRows;
    maxFeature = get_features(bl, ' ', ':', featureList);
    if ( maxFeature > nCols )
      nCols = maxFeature;
    nNonZeroes += (featureList.size()-1); // -1 for label
    featureList.clear();
  } // while

  if ( 0 == nRows )
    throw std::domain_error("empty file");

  std::size_t modRows = 0, oobRows = 0, oobRowCount = 0;
  DataMatrixInv* dmOOB = nullptr;
  if ( oob ) {
    oobRows = std::floor(oobPercent * nRows);
    if ( !oobRows )
      throw std::domain_error("--xval percentage * nRows of input rounds down to 0.  Give a bigger input or increase your --xval percentage");
    modRows = std::floor(nRows / oobRows);
    if ( !modRows )
      throw std::domain_error("something unexpected is wrong.  See 'read_data()s modRows'");

    // have to read through again to derive proper nNonZeroes and nNonZeroesOOB, and possibly update nCols
    infile.close();
    infile.open(source.c_str());
    std::size_t nNonZeroesOOB = 0;
    nRows = 0, nNonZeroes = 0, oobRowCount = 0;
    while ( infile >> bl ) {
      maxFeature = get_features(bl, ' ', ':', featureList);
      if ( !((nRows+1) % modRows) || oobRowCount >= oobRows ) {
        nNonZeroes += (featureList.size()-1);
        ++nRows;
      } else {
        nNonZeroesOOB += (featureList.size()-1);
        ++oobRowCount;
      }
      if ( maxFeature > nCols )
        nCols = maxFeature;
      featureList.clear();
    } // while
    // keep nCols the same between dmOOB and the learning matrix; nCols = finalNCols below
    dmOOB = new DataMatrixInv(oobRows, nCols, nNonZeroesOOB);
  }

  const std::size_t finalNCols = nCols;

  DataMatrix* dm = new DataMatrix(nRows, finalNCols, nNonZeroes);
  DataMatrix& dmref = *dm;
  infile.close();
  infile.open(source.c_str());
  nRows = 0, oobRowCount = 0;
  while ( infile >> bl ) {
    get_features(bl, ' ', ':', featureList);
    if ( !oob || !((nRows+1) % modRows) || dmOOB->size1() >= oobRows ) {
      for ( const auto& i : featureList )
        dmref(nRows, i.first) = i.second;
      ++nRows;
    } else { // dmOOB != nullptr
      auto& oobRef = *dmOOB;
      for ( const auto& i : featureList )
        oobRef(oobRowCount, i.first) = i.second;
      ++oobRowCount;
    }
    featureList.clear();
  } // while
  return std::make_tuple(dm, dmOOB, finalNCols);
}

} // namespace Tree

#endif // __DECISION_TREE_INPUT__
