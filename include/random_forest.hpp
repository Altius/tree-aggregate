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

#ifndef __ALTIUS_RANDFOREST_HPP__
#define __ALTIUS_RANDFOREST_HPP__

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

#include "dtree.hpp"
#include "utils.hpp"

namespace Forest {

struct RandomForest {

  RandomForest(Tree::DecisionTreeParameters& dtp, const Tree::DataMatrix& dm, Tree::DataMatrixInv const* oobs)
    : _dtp(dtp), _dm(dm), _oobs(oobs) {}

  void
  build_trees(std::size_t nTrees, Tree::DecisionTreeParameters& dtp) {
    for ( std::size_t idx = 0; idx < nTrees; ++idx ) {
      auto nextTree = new Tree::DecisionTreeClassifier(dtp);
// sjn use multithreading here when building trees
// need to separate uniform prng from dtp, make dtp const above and in DTC class, and in main()
// each thread receives its own prng with some unique offset from the given _seed
      nextTree->learn(_dm);
      _forest.push_back(nextTree);
    } // for
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
  const Tree::DataMatrix& _dm;
  Tree::DataMatrixInv const* _oobs;
  std::list<Tree::DecisionTreeClassifier*> _forest;
};

std::ostream&
operator<<(std::ostream& os, const RandomForest& rf) {
  for ( auto& a : rf._forest )
    os << "[" << std::endl << *a << "]" << std::endl;
  return os;
}

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

#endif // __ALTIUS_RANDFOREST_HPP__
