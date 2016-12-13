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

const featureID LabelID = -1;

struct Help {};

std::string Usage(const std::string& progName) {
  std::string msg;
  msg += progName + " --help or --version";
  msg += "\n" + progName + " learn-model [--njobs <+int>] [--xval <out-of-bag-percentage>] [--seed <+int>] [--nsplit-features <+int>] <number-trees> <labeled-features>";
  msg += "\n" + progName + " predict [--njobs <+integer>] <model> <new-features>";
  msg += "\n" + progName + " Most output goes to stdout.  The result of an optional cross-validation using --xval is sent to stderr.";
  return msg;
}

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
//    const DecisionTreeParameters& dtp = *input._dtp;
//    build_trees(input._nTrees, dtp, *data.first, *data.second);
//use multithreading
//    delete datapair.first;
//    delete datapair.second;
    return EXIT_SUCCESS;
  } catch(Help h) {
    std::cout << Usage(argv[0]) << std::endl;
    return EXIT_SUCCESS;
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
