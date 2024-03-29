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

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../include/dtree_helpers.hpp"
#include "../include/random_forest.hpp"
#include "../include/utils.hpp"

std::string Usage(const std::string& progName) {
  std::string msg;
  msg += progName + " --help or --version";
  msg += "\n" + progName + " learn [--njobs=<+int>] [--xval=<+real>] [--seed=<+int>] [--nsplit-features=<SQRT|LOG2|+int|+perc>]";
  msg += "\n                     [--min-purity=<+real>] [--split-strategy=<BEST|RANDOM>] <number-trees> <labeled-features>";
  msg += "\n" + progName + " predict [--njobs=<+integer>] [--probs] <model> <new-features>";
  msg += "\n" + progName + " Most output goes to stdout.  The result of an optional cross-validation using --xval is sent to stderr.";
  return msg;
}

struct Help {};
struct Version {};
struct NoInput {};

struct Input {
  Input(int argc, char** argv);

  enum class Mode { LEARN, PREDICT };

  Mode _mode;
  bool _oobTests;
  double _oobPercent;
  std::size_t _nJobs;
  std::size_t _nTrees;
  std::size_t _nFeatures;
  std::size_t _nLabels;
  std::string _modelFile;
  std::string _dataSourceLearn;
  std::string _dataSourcePredict;
  Tree::DecisionTreeParameters* _dtp;

  bool _probs;

  friend std::ostream&
  operator<<(std::ostream& os, const Input& input) {
    os << "number-jobs=" << input._nJobs;
    os << " " << "number-trees=" << input._nTrees;
    os << " " << "data-source=" << input._dataSourceLearn;
    os << " " << "number-features=" << input._nFeatures;
    os << " " << "max-label=" << input._nLabels;
    return os;
  }

  friend std::istream&
  operator>>(std::istream& is, Input& input) {
    input._mode = Mode::PREDICT;
    Utils::ByLine bl;
    if ( is >> bl ) {
      auto params = Utils::split(bl, " ");
      for ( auto& i : params )  {
        auto pval = Utils::split(i, "=");
        if ( pval.size()==2 ) {
          if ( pval[0] == "number-jobs" )
            input._nJobs = Utils::convert<decltype(input._nJobs)>(pval[1], Utils::Nums::PLUSINT_NOZERO);
          else if ( pval[0] == "number-trees" )
            input._nTrees = Utils::convert<decltype(input._nTrees)>(pval[1], Utils::Nums::PLUSINT_NOZERO);
          else if ( pval[0] == "data-source" )
            input._dataSourceLearn = pval[1];
          else if ( pval[0] == "number-features" )
            input._nFeatures = Utils::convert<decltype(input._nFeatures)>(pval[1], Utils::Nums::PLUSINT_NOZERO);
          else if ( pval[0] == "max-label" )
            input._nLabels = Utils::convert<decltype(input._nLabels)>(pval[1], Utils::Nums::PLUSINT_NOZERO);
        } else {
          throw std::domain_error("bad model file: Input Class");
        }
      } // for
    }
    return is;
  }

  ~Input();
};

Forest::RandomForest*
read_model(Input&);

int main(int argc, char** argv) {
  try {
    Input input(argc, argv);

    if ( input._mode == Input::Mode::LEARN ) {
// sjn probably need to make a _unif and _rng per thread (adding one to the core seed value) and pull it out of dtp so that it can be const (and randomforest class)

      // return 4-tuple of Training Data, Optional Testing Data, Max Feature ID, and Max Label
      constexpr std::size_t TRN  = 0;
      constexpr std::size_t TST  = 1;
      constexpr std::size_t MXFT = 2;
      constexpr std::size_t MXLB = 3;
      constexpr std::size_t CNTS = 4;
      auto data = Tree::read_data(input._dataSourceLearn, input._oobTests, input._oobPercent);

      Tree::DecisionTreeParameters& dtp = *input._dtp;
      dtp.set_weights_from_counts(std::get<CNTS>(data));

      input._nFeatures = static_cast<std::size_t>(std::get<MXFT>(data));
      input._nLabels = static_cast<std::size_t>(std::get<MXLB>(data));

      std::cout << input << std::endl;
      std::cout << dtp << std::endl;

      Forest::RandomForest rf(dtp, std::get<TRN>(data), std::get<TST>(data));
      rf.build_trees(input._nTrees);

      if ( std::get<TRN>(data) )
        delete std::get<TRN>(data);
      if ( std::get<TST>(data) )
        delete std::get<TST>(data);

/*
      if ( datapair.second ) { // x-validation
        auto stats = xval_estimates(rf._forest, *datapair.second);
        std::cerr << stats << std::endl;
      }
*/
    } else { // classify new features
      Forest::RandomForest* rf = read_model(input);
      std::ifstream infile(input._dataSourcePredict.c_str());
      std::istream_iterator<Utils::ByLine> start(infile), eos;
      if ( input._probs )
        rf->predict_row_probs(start, eos, input._nFeatures);
      else
//        rf->predict_row(start, eos, input._nFeatures);
// for challenge parallelization
rf->predict_row_per_tree(start, eos, input._nFeatures);

      delete rf;
    }

    return EXIT_SUCCESS;
  } catch(Help h) {
    std::cout << Usage(argv[0]) << std::endl;
    return EXIT_SUCCESS;
  } catch(Version v) {
    std::cout << Tree::FileFormatVersion << std::endl; // given in dtree.hpp
    return EXIT_SUCCESS;
  } catch(NoInput n) {
    std::cout << Usage(argv[0]) << std::endl;
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

Forest::RandomForest*
read_model(Input& input) {
  std::ifstream infile(input._modelFile.c_str());
  if ( !infile )
    throw std::domain_error("Unable to locate model file given: " + input._modelFile);

  infile >> input;
  infile >> *input._dtp;
  Forest::RandomForest* rf = new Forest::RandomForest(*input._dtp);
  infile >> *rf;
  return rf;
}

Forest::RandomForest*
read_model_in_chunks(Input& input, std::size_t chunks = 100) {
  std::ifstream infile(input._modelFile.c_str());
  if ( !infile )
    throw std::domain_error("Unable to locate model file given: " + input._modelFile);

  infile >> input;
  infile >> *input._dtp;
  Forest::RandomForest* rf = new Forest::RandomForest(*input._dtp, input._modelFile, chunks);
  return rf;
}

Input::Input(int argc, char** argv) : _mode(Mode::LEARN),
                                      _oobTests(false),
                                      _oobPercent(0),
                                      _nJobs(1),
                                      _nTrees(0),
                                      _dtp(nullptr),
                                      _probs(false) {
  using namespace Tree;

  for ( int i = 1; i < argc; ++i ) {
    if ( argv[i] == std::string("--help") )
      throw Help();
    else if ( argv[i] == std::string("--version") )
      throw Version();
  } // for

  if ( argc < 3 ) {
    if ( argc == 1 )
      throw NoInput();
    throw std::domain_error("Wrong # args");
  }

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
  dtype minPurity = 1; // no limit at 1
  bool useZeroes = true;
  ClassWeightType classWeight = ClassWeightType::BALANCED;
  std::size_t minPerClassSampleSize = 1;
  uint64_t seed = 0; // highly randomized if not set

  bool nonopt = false;
  int numnonopt = 0;
  for ( int i = 2; i < argc; ++i ) {
    auto next = Utils::split(argv[i], "="); // function from dtree.hpp
    if ( next.size() != 2 ) {
      // avoid nonopt checks for boolean options
      if ( next[0] == "--probs" ) {
        _probs = true;
        continue;
      }

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
          _dataSourceLearn = next[0];
      } else { // _mode == Mode::PREDICT
        if ( numnonopt > 2 )
          throw std::domain_error("Too many non-option args");
        else if ( numnonopt == 1 )
          _modelFile = next[0];
        else
          _dataSourcePredict = next[0];
      }
    } else {
      if ( nonopt ) // options should come before required file inputs
        throw std::domain_error("option " + next[0] + " given after a non-option.");

      if ( next[0] == "--njobs" )
        _nJobs = Utils::convert<decltype(_nJobs)>(next[1], Utils::Nums::PLUSINT);
      else { // option must be specific to _mode == Mode::LEARN
        if ( _mode != Mode::LEARN )
          throw std::domain_error(next[0] + " does not make sense when not in learn mode");

        if ( next[0] == "--xval" ) {
          _oobTests = true;
          _oobPercent = Utils::convert<decltype(_oobPercent)>(next[1], Utils::Nums::PLUSREAL);
          if ( _oobPercent < 0.01 || _oobPercent > 0.49 )
            throw std::domain_error("--xval should have a value b/n 0.01 and 0.49");
        }
        else if ( next[0] == "--seed" ) {
          seed = Utils::convert<decltype(seed)>(next[1], Utils::Nums::PLUSBIGINT);
        }
        else if ( next[0] == "--split-strategy" ) {
          if ( Utils::uppercase(next[1]) == "RANDOM" )
            strategy = SplitStrategy::RANDOM;
          else if ( Utils::uppercase(next[1]) != "BEST" ) // BEST is default
            throw std::domain_error("--split-strategy must be set to random or best");
        }
        else if ( next[0] == "--nsplit-features" ) {
          if ( Utils::uppercase(next[1]) == "SQRT" )
            splitMaxFeat = SplitMaxFeat::SQRT;
          else if ( Utils::uppercase(next[1]) == "LOG2" )
            splitMaxFeat = SplitMaxFeat::LOG2;
          else { // INT or FLT
            try {
              int dummy = Utils::convert<int>(next[1], Utils::Nums::PLUSINT_NOZERO);
              splitMaxFeat = SplitMaxFeat::INT;
              splitVal = dummy;
            } catch(std::domain_error de) {
              dtype dummy = Utils::convert<dtype>(next[1], Utils::Nums::PLUSREAL_NOZERO);
              if ( dummy < 0.01 || dummy > 1 )
                throw std::domain_error("require " + next[0] + " value: 0.01 <= value <= 1");
              splitMaxFeat = SplitMaxFeat::FLT;
              splitVal = dummy;
            }
          }
        } else if ( next[0] == "--min-purity" ) {
          dtype dummy = Utils::convert<dtype>(next[1], Utils::Nums::PLUSREAL_NOZERO);
          if ( dummy < 0.01 || dummy > 1 )
            throw std::domain_error("require " + next[0] + " value: 0.01 <= value <= 1");
          minPurity = dummy;
        }
        else
          throw std::domain_error("unknown option: " + next[0]);
      }
    }
  } // for

  if ( _mode != Mode::PREDICT && _probs )
    throw std::domain_error("--probs only works with mode 'predict'");

  _dtp = new Tree::DecisionTreeParameters(
                                    criterion, strategy, splitMaxFeat, splitVal,
                                    mxDepth, mxLeaves, minLeafSamples, minPerClassSampleSize,
                                    minPurity, useZeroes, classWeight, seed
                                         );
}

Input::~Input() {
  if ( _dtp )
    delete _dtp;
}
