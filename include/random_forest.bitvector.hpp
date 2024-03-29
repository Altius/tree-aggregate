/*
  FILE: random_forest.hpp
  AUTHOR: Shane Neph
  CREATE DATE: Dec.2016
*/

//
//    Random Forest
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

#ifndef __ALTIUS_RANDFOREST_BITVECTOR_HPP__
#define __ALTIUS_RANDFOREST_BITVECTOR_HPP__

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dtree.bitvector.hpp"
#include "utils.hpp"

namespace Forest {

struct RandomForest {

  // when learning
  RandomForest(Tree::DecisionTreeParameters& dtp, Tree::DataMatrix const* dm, Tree::DataMatrixInv const* oobs)
    : _dtp(dtp), _dm(dm), _oobs(oobs) {}

  // when predicting
  explicit RandomForest(Tree::DecisionTreeParameters& dtp)
    : _dtp(dtp), _dm(nullptr), _oobs(nullptr) {}

  void
  build_trees(std::size_t nTrees) {
    for ( std::size_t idx = 0; idx < nTrees; ++idx ) {
      auto nextTree = new Tree::DecisionTreeClassifier(_dtp);
// sjn use multithreading here when building trees
// need to separate uniform prng from dtp, make dtp const above and in DTC class, and in main()
// each thread receives its own prng with some unique offset from the given _seed
      nextTree->learn(*_dm);
      // _forest.push_back(nextTree);
      delete nextTree;
    } // for
  }
/*
  std::vector<Tree::label>
  predict(Tree::DataMatrixInv const* data, std::size_t mxlabel) const {
// sjn multithread this
    std::vector<std::vector<Tree::label>> results(data->size1(), std::vector<Tree::label>(mxlabel, (Tree::label)0));
    for ( auto& tree : _forest ) {
      std::vector<Tree::label> predictions(data->size1(), 0);
      tree->classify(*data, predictions);
      for ( std::size_t i = 0; i < predictions.size(); ++i )
        results[i][predictions[i]]++;
    } // for

    std::vector<Tree::label> final(data->size1(), (Tree::label)0);
    for ( std::size_t i = 0; i < results.size(); ++i )
      final[i] = static_cast<Tree::label>(std::max_element(results[i].begin(), results[i].end()) - results[i].begin());
    return final;
  }
*/

  std::vector<Tree::label>
  predict(Tree::DataMatrixInv const* data, std::size_t mxlabel) const {
// sjn multithread this
    std::vector<std::vector<Tree::label>> results(data->size1(), std::vector<Tree::label>(mxlabel, (Tree::label)0));
    for ( auto& tree : _forest ) {
      std::vector<Tree::label> predictions(data->size1(), 0);
      tree->classify(*data, predictions);
      for ( std::size_t i = 0; i < predictions.size(); ++i )
        results[i][predictions[i]]++;
    } // for

    std::vector<Tree::label> final(data->size1(), (Tree::label)0);
    for ( std::size_t i = 0; i < results.size(); ++i )
      final[i] = static_cast<Tree::label>(std::max_element(results[i].begin(), results[i].end()) - results[i].begin());
    return final;
  }

  // implemented for other machine learning techniques to use Decision Tree outputs as features
  template <typename Initer>
  void
  predict_row_per_tree(Initer start, Initer end, std::size_t maxFeat) const {
    while ( start != end ) {
      auto w = *start++;
      bool first = false;
      for ( auto& tree : _forest ) {
        auto val = tree->classify(w, maxFeat);
        if ( !first )
          std::cout << " ";
        std::cout << val;
        first = false;
      } // for
      std::cout << std::endl;
    } // while
  }

  template <typename Initer>
  void
  predict_row(Initer start, Initer end, std::size_t maxFeat) const {
    while ( start != end ) {
      auto w = *start++;
      std::vector<Tree::label> results(_forest.size(), Tree::label(0));
      for ( auto& tree : _forest )
        results[tree->classify(w, maxFeat)] += 1;
      std::cout << static_cast<Tree::label>(std::max_element(results.begin(), results.end()) - results.begin());
      std::cout << std::endl;
    } // while
  }

  template <typename Initer>
  void
  predict_row_probs(Initer start, Initer end, std::size_t maxFeat) const {
    while ( start != end ) {
      std::string value = *start++;
      std::map<Tree::label, std::size_t> counts;
      for ( auto& tree : _forest )
        counts[tree->classify(value, maxFeat)] += 1; // size_t guaranteed to be 0 by default?
      double sum = 0.;
      for ( auto& c : counts )
        sum += c.second;
      for ( auto& c : counts )
        c.second /= sum;
      std::cout << "(";
      auto iter = counts.begin();
      if ( iter != counts.end() ) {
        std::cout << iter->second;
        while ( ++iter != counts.end() )
          std::cout << "," << iter->second;
      }
      std::cout << ")" << std::endl;
      ++start;
    } // while
  }

  double
  variable_importance() const {
    return 0;
  }

  double
  cross_validate() const {
    return 0;
  }

  ~RandomForest() {
    for ( auto ntree : _forest )
      delete ntree;
  }

  friend std::ostream&
  operator<<(std::ostream&, const RandomForest&);

  friend std::istream&
  operator>>(std::istream&, RandomForest&);

  Tree::DecisionTreeParameters& _dtp;
  Tree::DataMatrix const* _dm;
  Tree::DataMatrixInv const* _oobs;
  std::list<Tree::DecisionTreeClassifier*> _forest;
};

inline
std::ostream&
operator<<(std::ostream& os, const RandomForest& rf) {
  bool any = false;
  for ( auto& tree : rf._forest ) {
    if ( any )
      os << std::endl;
    os << *tree;
    any = true;
  } // for
  return os;
}

inline
std::istream&
operator>>(std::istream& is, RandomForest& rf) {
  Tree::DecisionTreeClassifier tmp(rf._dtp);
  while ( is >> tmp ) {
    rf._forest.push_back(new Tree::DecisionTreeClassifier(tmp));
    tmp._nodes.clear();
  } // while
  return is;
}

} // namespace Forest

#endif // __ALTIUS_RANDFOREST_BITVECTOR_HPP__
