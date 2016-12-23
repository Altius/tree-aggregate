/*
  FILE: random_forest.cpp
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

#include "../include/dtree.hpp"
#include "../include/dtree-input.hpp"
#include "../include/utils.hpp"

using namespace Tree;

std::string Usage(const std::string& progName) {
  std::string msg;
  msg += progName + " --help or --version";
  msg += "\n" + progName + " learn [--njobs=<+int>] [--xval=<out-of-bag-percentage>] [--seed=<+int>] [--nsplit-features=<SQRT|LOG2|+int|+perc>] <number-trees> <labeled-features>";
  msg += "\n" + progName + " predict [--njobs=<+integer>] <model> <new-features>";
  msg += "\n" + progName + " Most output goes to stdout.  The result of an optional cross-validation using --xval is sent to stderr.";
  return msg;
}

struct Help {};
struct Version {};

struct Input {
  Input(int argc, char** argv);

  enum class Mode { LEARN, PREDICT };

  Mode _mode;
  bool _saveModels;
  bool _oobTests;
  double _oobPercent;
  std::size_t _nJobs;
  std::size_t _nTrees;
  std::string _modelFile;
  std::string _dataSource;
  DecisionTreeParameters* _dtp;

  ~Input();
};

struct RandomForest {
  RandomForest(const DataMatrix& dm, const DataMatrixInv& oobs) : _dm(dm), _oobs(oobs) {}

  const DataMatrix& _dm;
  const DataMatrixInv& _oobs;
  std::list<DecisionTreeClassifier*> _forest;
};

double GetVariableImportance(const RandomForest& rf) {
return 0;
}

void
build_trees(std::size_t nTrees, DecisionTreeParameters& dtp, const DataMatrix& train, std::list<DecisionTreeClassifier*>& trees) {
  for ( std::size_t idx = 0; idx < nTrees; ++idx ) {
    auto nextTree = new DecisionTreeClassifier(dtp);
// sjn use multithreading here when building trees
// need to separate uniform prng from dtp, make dtp const above and in DTC class, and in main()
// each thread receives its own prng with some unique offset from the given _seed
    nextTree->learn(train);
    trees.push_back(nextTree);
  } // for
}


int main(int argc, char** argv) {
  try {
    Input input(argc, argv);
// sjn probably need to make a _unif and _rng per thread (adding one to the core seed value) and pull it out of dtp so that it can be const
    auto datapair = read_data(input._dataSource, input._oobTests, input._oobPercent);

    DecisionTreeParameters& dtp = *input._dtp;
    std::list<DecisionTreeClassifier*> dtcs;
    if ( input._mode == Input::Mode::LEARN ) {
      build_trees(input._nTrees, dtp, *datapair.first, dtcs);
//      if ( datapair.second ) // x-validation
//        auto stats = xval_estimates(dtcs, *datapair.second);
//      if ( input._saveModels )
//        save_models(dtcs, input._modelFile, input); // need to save input data as first line
    } else { // classify new features
//      read_models(input._modelFile, dtcs);
//      classifications(classify(*datapair.first, dtcs);
    }

    if ( datapair.first )
      delete datapair.first;
    if ( datapair.second )
      delete datapair.second;
    for ( auto ntree : dtcs )
      delete ntree;
    return EXIT_SUCCESS;
  } catch(Help h) {
    std::cout << Usage(argv[0]) << std::endl;
    return EXIT_SUCCESS;
  } catch(Version v) {
    std::cout << FileFormatVersion << std::endl; // given in dtree.hpp
  } catch(char const* c) {
    std::cerr << c << std::endl;
  } catch(std::string& s) {
    std::cerr << s << std::endl;
  } catch(std::exception& e) {
    std::cerr << e.what() << std::endl;
  } catch(...) {
    std::cerr << "Unknown failure" << std::endl;
  }
  return EXIT_FAILURE;
}

Input::Input(int argc, char** argv) : _mode(Mode::LEARN),
                                      _saveModels(false),
                                      _oobTests(false),
                                      _oobPercent(0),
                                      _nJobs(1),
                                      _nTrees(0),
                                      _dtp(nullptr) {
  for ( int i = 1; i < argc; ++i ) {
    if ( argv[i] == std::string("--help") )
      throw(Help());
    else if ( argv[i] == std::string("--version") )
      throw(Version());
  } // for

  if ( argc < 3 )
    throw std::domain_error("Wrong # args");

  std::string nxt = argv[1];
  if ( nxt == "predict" )
    _mode = Mode::PREDICT;
  else if ( nxt == "learn" )
    _mode = Mode::LEARN;
  else
    throw std::domain_error("unknown mode: " + nxt);

  SplitMaxFeat splitMaxFeat = SplitMaxFeat::SQRT;
  dtype splitVal = 0;
  auto criterion = Criterion::GINI;
  auto strategy = SplitStrategy::BEST;
  int mxDepth = 0; // no limit at 0
  int mxLeaves = 0; // no limit at 0
  int minLeafSamples = 1; // no limit at 1
  dtype minPurity = 0.7; // no limit at 0
  bool useZeroes = false;
  ClassWeightType classWeight = ClassWeightType::SAME;
  uint64_t seed = 0; // highly randomized if not set

  bool nonopt = false;
  int numnonopt = 0;
  for ( int i = 2; i < argc; ++i ) {
    auto next = Utils::split(argv[i], "="); // function from dtree.hpp
    if ( next.size() != 2 ) {
      if ( next.size() != 1 )
        throw std::domain_error("Bad argument: " + std::string(argv[i]));
      nonopt = true;
      ++numnonopt;
      if ( _mode == Mode::LEARN ) {
        if ( numnonopt > 2 )
          throw std::domain_error("Too many non-option args");
        else if ( numnonopt == 1 )
          _nTrees = Utils::convert<decltype(_nTrees)>(next[0], Utils::Nums::PLUSINT_NOZERO);
        else
          _dataSource = next[0];
      } else { // _mode == Mode::PREDICT
        if ( numnonopt > 2 )
          throw std::domain_error("Too many non-option args");
        else if ( numnonopt == 1 )
          _modelFile = next[0];
        else
          _dataSource = next[0];
      }
    } else {
      if ( nonopt ) // options should come before required file inputs
        throw std::domain_error("option " + next[0] + " given after a non-option.");

      if ( next[0] == "--njobs" )
        _nJobs = Utils::convert<decltype(_nJobs)>(next[1], Utils::Nums::PLUSINT);
      else { // specific to _mode == Mode::LEARN
        if ( _mode != Mode::LEARN )
          throw std::domain_error(next[0] + " does not make sense when not in learn-model mode");

        if ( next[0] == "--xval" ) {
          _oobTests = true;
          _oobPercent = Utils::convert<decltype(_oobPercent)>(next[1], Utils::Nums::PLUSREAL);
          if ( _oobPercent < 0.01 || _oobPercent > 0.49 )
            throw std::domain_error("--xval should have a value b/n 0.01 and 0.49");
        }
        else if ( next[0] == "--seed" ) {
          seed = Utils::convert<decltype(seed)>(next[1], Utils::Nums::PLUSBIGINT);
          throw std::domain_error("--seed does not make sense when not in learn-model mode");
        }
        else if ( next[0] == "--nsplit-features" ) {
          if ( _mode != Mode::LEARN )
            throw std::domain_error("--nsplit-features does not make sense when not in learn-model mode");
  
          if ( Utils::uppercase(next[1]) == "SQRT" )
            splitMaxFeat = SplitMaxFeat::SQRT;
          else if ( Utils::uppercase(next[1]) == "LOG2" )
            splitMaxFeat = SplitMaxFeat::LOG2;
          else { // INT or FLT
            try {
              int dummy = Utils::convert<int>(next[1], Utils::Nums::PLUSINT_NOZERO);
              splitVal = dummy;
            } catch(std::domain_error de) {
              double dummy = Utils::convert<double>(next[1], Utils::Nums::PLUSREAL);
              if ( dummy < 0.01 || dummy > 1 )
                throw std::domain_error("require " + next[0] + " value: 0.01 <= value <= 1");
              splitVal = dummy;
            }
          }
        }
        else
          throw std::domain_error("unknown option: " + next[0]);
      }
    }
  } // for

  _dtp = new DecisionTreeParameters(
                                    criterion, strategy, splitMaxFeat, splitVal,
                                    mxDepth, mxLeaves, minLeafSamples, minPurity,
                                    useZeroes, classWeight, seed
                                   );
}

Input::~Input() {
  if ( _dtp )
    delete _dtp;
}
