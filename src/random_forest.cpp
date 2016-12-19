#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include "../include/dtree.hpp"
#include "../include/dtree-input.hpp"

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

  enum Mode { LEARN, PREDICT };

  Mode _mode;
  bool _saveModels;
  bool _oobTests;
  double _oobPercent;
  std::size_t _nJobs;
  std::size_t _nTrees;
  std::string _dataSource;
  DecisionTreeParameters* _dtp;

  ~Input();
};

struct RandomForest {
  DataMatrix& _dm;
  DataMatrixInv& _oobs;
  std::list<DecisionTreeClassifier> _forest;
};

double GetVariableImportance(const RandomForest& rf) {
return 0;
}


int main(int argc, char** argv) {
  try {
    Input input(argc, argv);
    auto datapair = read_data(input._dataSource, input._oobTests, input._oobPercent); // _dataSource includes labels
//std::cout << *datapair.first << std::endl;

    const DecisionTreeParameters& dtp = *input._dtp;
//    build_trees(input._nTrees, dtp, *datapair.first, *datapair.second);
//use multithreading
    if ( datapair.first )
      delete datapair.first;
    if ( datapair.second )
      delete datapair.second;
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



Input::Input(int argc, char** argv) : _mode(LEARN),
                                      _saveModels(false),
                                      _oobTests(false),
                                      _oobPercent(0),
                                      _nJobs(1),
                                      _nTrees(0),
                                      _dtp(nullptr) {
  if ( argc < 3 )
    throw std::domain_error("Wrong # args");

  for ( int i = 1; i < argc; ++i ) {
    if ( argv[i] == std::string("--help") )
      throw(Help());
    else if ( argv[i] == std::string("--version") )
      throw(Version());
  } // for

  std::string nxt = argv[1];
  if ( nxt == "predict" )
    _mode = PREDICT;
  else if ( nxt == "learn" )
    _mode = LEARN;
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
    auto narg = split(argv[i], "="); // from dtree.hpp
    if ( narg.size() != 2 ) {
      if ( narg.size() != 1 )
        throw std::domain_error("Bad argument: " + std::string(argv[i]));
      nonopt = true;
      ++numnonopt;
      if ( _mode == LEARN ) {
        if ( numnonopt > 2 )
          throw std::domain_error("Too many non-option args");
      } else { // _mode == PREDICT
        if ( numnonopt > 1 )
          throw std::domain_error("Too many non-option args");
      }
    } else {
      if ( nonopt )
        throw std::domain_error("option given after non-option " + narg[0]);

      if ( narg[0] == "--njobs" )
        _nJobs = convert<decltype(_nJobs)>(next[1], plusint);
      else { // specific to _mode == LEARN
        if ( _mode != LEARN )
          throw std::domain-error(narg[0] + " does not make sense when not in learn-model mode");

        if ( narg[0] == "--xval" ) {
          _oobTests = true;
          _oobPercent = convert<decltype(_oobPercent)>(next[1], plusreal);
          if ( _oobPercent < 0.01 || _oobPercent > 0.49 )
            throw std::domain_error("--xval should have a value b/n 0.01 and 0.49");
        }
        else if ( narg[0] == "--seed" ) {
          seed = convert<decltype(seed)>(next[1], bigplusinteger);
          throw std::domain-error("--seed does not make sense when not in learn-model mode");
        }
        else if ( narg[0] == "--nsplit-features" ) {
          if ( _mode != LEARN )
            throw std::domain-error("--nsplit-features does not make sense when not in learn-model mode");
  
          if ( toupper(narg[1]) == "SQRT" )
            splitMaxFeat = SplitMaxFeat::SQRT;
          else if ( toupper(narg[1]) == "LOG2" )
            splitMaxFeat = SplitMaxFeat::LOG2;
          else { // INT or FLT
            try {
              int dummy = convert<int>(next[1], plusint);
              splitVal = dummy;
            } catch(std::domain_error de) {
              double dummy = convert<double>(next[1], plusreal);
              if ( dummy < 0.01 || dummy > 1 )
                throw std::domain_error("require " + narg[0] + " value: 0.01 <= value <= 1");
              splitVal = dummy;
            }
          }
          else
            throw std::domain_error("problem with --nsplit-features=" + narg[1] + ". See --help");
        }
        else
          throw std::domain_error("unknown option: " + narg[0]);
      }
    }
  } // for

  _dtp = new DecisionTreeParameters(criterion, strategy, splitMaxFeat, splitVal,
                                    mxDepth, mxLeaves, minLeafSamples, minPurity,
                                    useZeroes, classWeight, seed
                                   );
}

Input::~Input() {
  if ( _dtp )
    delete _dtp;
}
