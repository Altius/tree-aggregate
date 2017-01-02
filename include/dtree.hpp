/*
  FILE: dtree.hpp
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

#ifndef __ALTIUS_DTREE_HPP__
#define __ALTIUS_DTREE_HPP__

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include "utils.hpp"

namespace Tree {

namespace ub = boost::numeric::ublas;

std::string FileFormatVersion = "0.1";

typedef unsigned short featureID;
typedef featureID label;
typedef float dtype;
typedef boost::dynamic_bitset<> Monitor;

template <typename T>
using ColumnVector = ub::vector<T>;
typedef ub::compressed_vector<dtype> RowVector;
typedef ColumnVector<RowVector> DataVectorVector;
typedef ub::generalized_vector_of_vector<dtype, ub::column_major, DataVectorVector> DataMatrix;
typedef ub::generalized_vector_of_vector<dtype, ub::row_major, DataVectorVector> DataMatrixInv;

enum struct Criterion { GINI, ENTROPY };
enum struct SplitStrategy { BEST, RANDOM };
enum struct SplitMaxFeat { INT, FLT, SQRT, LOG2 };
enum struct ClassWeightType { SAME, BALANCED };

/* Working under the assumption that decision trees will use the same criteria and such
     through a random forest or through svm feature learning on the decision tree decisions.
     This assumption helps improve memory use.
   All decision trees will use the same DecisionTreeParameters
*/
struct DecisionTreeClassifier;

struct DecisionTreeParameters {
  template <typename SplitType>
  explicit DecisionTreeParameters(
                         Criterion c = Criterion::GINI,
                         SplitStrategy s = SplitStrategy::BEST,
                         SplitMaxFeat m = SplitMaxFeat::SQRT,
                         SplitType splitValue = SplitType(),
                         std::size_t maxDepth = 0,
                         std::size_t maxLeafNodes = 0,
                         std::size_t minSamplesLeaf = 1,
                         std::size_t minClassSamples = 0,
                         dtype minPuritySplit = 0.7,
                         bool useImpliedZeroesInSplitDecision = true,
                         ClassWeightType cw = ClassWeightType::SAME,
                         uint64_t randomState = 0
                                 )
                  : _criterion(c),
                    _splitStrategy(s),
                    _splitMaxSelect(m),
                    _maxDepth(maxDepth),
                    _maxLeafNodes(maxLeafNodes),
                    _minLeafSamples(minSamplesLeaf),
                    _minClassSamples(minClassSamples),
                    _minPuritySplit(minPuritySplit),
                    _useImpliedZerosInSplitDecision(useImpliedZeroesInSplitDecision),
                    _classWeightType(cw),
                    _seed(randomState),
                    _selectSplitNumeric(splitValue),
                    _unif(0,1)
    {
      if ( 0 == _seed )
        _seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      do_seed();

      switch(_splitMaxSelect) {
        case SplitMaxFeat::INT:
          if ( _selectSplitNumeric < 1 )
            throw std::domain_error("SplitMaxFeat::INT selected, but value given is < 1");
          _selectSplitNumeric = std::round(_selectSplitNumeric);
          break;
        case SplitMaxFeat::FLT:
          if ( _selectSplitNumeric <= 0 )
            throw std::domain_error("SplitMaxFeat::FLT selected, but value given is <= 0. Expect: 0 < x <= 1");
          else if ( _selectSplitNumeric > 1 )
            throw std::domain_error("SplitMaxFeat::FLT selected, but value given is > 1.  Expect: 0 < x <= 1");
          break;
        default:
          break;
      }

      if ( _minPuritySplit <= 0.5 )
        throw std::domain_error("minimum purity setting must be > 0.5");

      _enums[Utils::uppercase("Criterion")] = static_cast<int>(_criterion);
      _enums[Utils::uppercase("SplitStrategy")] = static_cast<int>(_splitStrategy);
      _enums[Utils::uppercase("SplitMaxFeat")] = static_cast<int>(_splitMaxSelect);
      _enums[Utils::uppercase("ClassWeightType")] = static_cast<int>(_classWeightType);

      _argmap[Utils::uppercase("MaxDepth")] = Utils::Nums::PLUSINT; // zero means unused; grow full
      _argmap[Utils::uppercase("MaxLeafNodes")] = Utils::Nums::PLUSINT; // zero means unused
      _argmap[Utils::uppercase("MinLeafSamples")] = Utils::Nums::PLUSREAL; // when set, split until Node has this number or fewer; FLT means % of #rows
      _argmap[Utils::uppercase("MinClassSamples")] = Utils::Nums::PLUSINT; // when set, stop splitting if any class drops below this value
      _argmap[Utils::uppercase("MinPuritySplit")] = Utils::Nums::PLUSREAL_NOZERO; // split if purity less than this
      _argmap[Utils::uppercase("UseZeroesInSplit")] = Utils::Nums::BOOL; // use missing data == zero ?
      _argmap[Utils::uppercase("Seed")] = Utils::Nums::PLUSINT; // zero means use random seed
      _argmap[Utils::uppercase("SelectSplitNumeric")] = Utils::Nums::PLUSREAL; // if SplitMaxFeatures=INT||FLT, this is the value
      _argmap[Utils::uppercase("ClassWeights")] = Utils::Nums::PLUSREAL;
    }

    void set_weights_from_counts(const std::vector<std::size_t>& counts) { // for balanced labels
      if ( counts.size() != 2 )
        throw std::domain_error("class only supports binary classification at the moment");
      else if ( std::count(counts.begin(), counts.end(), 0) != 0 )
        throw std::domain_error("counts for a class label are 0");
      dtype total = std::accumulate(counts.begin(), counts.end(), 0);
      _weights[0] = total/(2*counts[0]);
      _weights[1] = total/(2*counts[1]);
    }

  friend inline
  std::ostream&
  operator<<(std::ostream& output, const DecisionTreeParameters& dtp) {
    constexpr char space = ' ';
    output << "FileFormat=" << FileFormatVersion;
    output << space << "Criterion=" << int(dtp._criterion);
    output << space << "SplitStrategy=" << int(dtp._splitStrategy);
    output << space << "SplitMaxFeat=" << int(dtp._splitMaxSelect);
    output << space << "MaxDepth=" << dtp._maxDepth;
    output << space << "MaxLeafNodes=" << dtp._maxLeafNodes;
    output << space << "MinLeafSamples=" << dtp._minLeafSamples;
    output << space << "MinClassSamples=" << dtp._minClassSamples;
    output << space << "MinPuritySplit=" << dtp._minPuritySplit;
    output << space << "UseZeroesInSplit=" << dtp._useImpliedZerosInSplitDecision;
    output << space << "ClassWeightType=" << int(dtp._classWeightType);
    output << space << "Seed=" << dtp._seed;
    output << space << "SelectSplitNumeric=" << dtp._selectSplitNumeric;
    output << space << "ClassWeights={" << dtp._weights[0];
    for ( std::size_t i = 1; i < dtp._weights.size(); ++i )
      output << "," << dtp._weights[i];
    output << "}";
    return output;
  }

  friend inline
  std::istream&
  operator>>(std::istream& is, DecisionTreeParameters& dtp) {
    const std::string space(" ");
    using Utils::uppercase;
    Utils::ByLine bl;
    is >> bl;
    auto vec = Utils::split(Utils::uppercase(bl), space);
    for ( auto& v : vec ) {
      auto jev = Utils::split(v, "=");
      if ( dtp._enums.find(jev[0]) != dtp._enums.end() ) {
        if ( jev[0] == Utils::uppercase("Criterion") ) {
          dtp._criterion = static_cast<Criterion>(Utils::convert<int>(jev[1], Utils::Nums::INT));
        } else if ( jev[0] == Utils::uppercase("SplitStrategy") ) {
          dtp._splitStrategy = static_cast<SplitStrategy>(Utils::convert<int>(jev[1], Utils::Nums::INT));
        } else if ( jev[0] == Utils::uppercase("SPLITMAXFEAT") ) {
          dtp._splitMaxSelect = static_cast<SplitMaxFeat>(Utils::convert<int>(jev[1], Utils::Nums::INT));
        } else if ( jev[0] == Utils::uppercase("ClassWeightType") ) {
          dtp._classWeightType = static_cast<ClassWeightType>(Utils::convert<int>(jev[1], Utils::Nums::INT));
        } else {
          throw std::domain_error("Programming error: operator>> for DecisionTreeParameters");
        }
      } else {
        if ( jev[0] == Utils::uppercase("FileFormat") )
          FileFormatVersion = jev[1];
        else if ( jev[0] == Utils::uppercase("MaxDepth") )
          dtp.set_value(dtp._maxDepth, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("MaxLeafNodes") )
          dtp.set_value(dtp._maxLeafNodes, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("MinLeafSamples") )
          dtp.set_value(dtp._minLeafSamples, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("MinClassSamples") )
          dtp.set_value(dtp._minClassSamples, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("MinPuritySplit") )
          dtp.set_value(dtp._minPuritySplit, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("UseZeroesInSplit") )
          dtp.set_value(dtp._useImpliedZerosInSplitDecision, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("Seed") )
          dtp.set_value(dtp._seed, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("SelectSplitNumeric") )
          dtp.set_value(dtp._selectSplitNumeric, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("ClassWeights") ) {
          jev[1] = jev[1].substr(0,jev[1].size()-1);
          jev[1] = jev[1].substr(1);
          auto kev = Utils::split(jev[1], ",");
          dtp.set_value(dtp._weights[0], jev[0], kev[0]);
          dtp.set_value(dtp._weights[1], jev[0], kev[1]);
        }
        else
          throw std::domain_error("model error: operator>> for DecisionTreeParameters: " + v);
      }
    } // for

    dtp._enums[Utils::uppercase("Criterion")] = dtp.get_enum(dtp._criterion);
    dtp._enums[Utils::uppercase("SplitStrategy")] = dtp.get_enum(dtp._splitStrategy);
    dtp._enums[Utils::uppercase("SplitMaxFeat")] = dtp.get_enum(dtp._splitMaxSelect);
    dtp._enums[Utils::uppercase("ClassWeightType")] = dtp.get_enum(dtp._classWeightType);

    dtp.do_seed();

    return is;
  }

private:
  template <typename T>
  T
  check_value(const std::string& s, const std::string& v) {
    auto f = _argmap.find(Utils::uppercase(s));
    if ( f == _argmap.end() )
      throw std::domain_error("Unknown Option: " + s);
    return Utils::convert<T>(v, f->second);
  }

  void do_seed() {
    std::seed_seq ss{uint32_t(_seed & 0xffffffff), uint32_t(_seed>>32)};
    _rng.seed(ss);
  }

  template <typename T>
  int get_enum(T t) { return static_cast<int>(t); }

  template <typename T>
  void set_value(T& t, const std::string& s, const std::string& v) { t = check_value<T>(s,v); }

// sjn private: for now public
public:
  friend struct DecisionTreeClassifier;

  // by construction (would be const if not for operator>> and my laziness to do it differently)
  Criterion _criterion;
  SplitStrategy _splitStrategy;
  SplitMaxFeat _splitMaxSelect; // if INT, then _selectSplitNumeric > 0.  if FLT, then 0 < _selectSplitNumeric <= 1
  std::size_t _maxDepth; // 0 means expand until all leaves meet purity criterion or until all nodes contain less than _minLeafSamples
  std::size_t _maxLeafNodes; // 0, then unlimited leaf nodes, else grow a tree until you reach _maxLeafNodes using bfs
  std::size_t _minLeafSamples; // 0, then not a stopping condition, else done splitting if #samples in leaf drops to this level
  std::size_t _minClassSamples; // 0, then not a stopping condition, else done splitting if any class drops below this value
  dtype _minPuritySplit; // A node will split if its purity is less than the threshold, otherwise it is a leaf
  bool _useImpliedZerosInSplitDecision;
  ClassWeightType _classWeightType;
  long int _seed;

  // set after or while data are read
  dtype _selectSplitNumeric; // if _splitMaxSelect is INT,FLT then this is the value of the INT or FLT
  std::array<dtype, 2> _weights;

  std::uniform_real_distribution<double> _unif;
  std::mt19937_64 _rng;

  // not user-settable
  std::unordered_map<std::string, int> _enums; // support operator>>
  std::unordered_map<std::string, Utils::Nums> _argmap; // support operator>>
};

template <typename SparseMatrix>
inline bool
allZeroes(const SparseMatrix& m, std::size_t featureCol) {
  // see http://stackoverflow.com/questions/7501996/accessing-the-indices-of-non-zero-elements-in-a-ublas-sparse-vector
  const auto& c = column(m, featureCol);
  return c.begin() == c.end();
}

template <typename SparseMatrix>
inline bool
allZeroes(const SparseMatrix& m, std::size_t featureCol, Monitor mask) {
  const auto& c = column(m, featureCol);
  if ( c.begin() == c.end() )
    return true;

  // see if any non-zero is an unmasked row
  for ( auto iter = c.begin(); iter != c.end(); ++iter ) {
    if ( mask[iter.index()] ) // unmasked and nonzero
      return false;
  } // for
  return true;
}

/*
  Hardcoding binary classification for now
*/
inline std::tuple<std::size_t, dtype>
purity(const std::array<dtype, 2>& a, const std::array<dtype, 2>& w) {
  if ( 0 == a[0] && 0 == a[1] )
    return std::make_tuple(0,1);
  return std::make_tuple(a[0]*w[0]>a[1]*w[1] ? 0 : 1, std::max(a[0]*w[0], a[1]*w[1])/(a[0]*w[0]+a[1]*w[1]));
}

inline label
weighted_vote_majority(const DataMatrix& dm, const Monitor& m, const DecisionTreeParameters& dtp) {
  std::vector<dtype> call;
  auto pos = m.find_first();
  auto& labels = column(dm, 0); // 0-based
  while ( pos != Monitor::npos ) {
    label labval = labels[pos];
    if ( call.size() <= labval )
      call.resize(static_cast<std::size_t>(labval+1), (label)0);
    call[labval] += 1*dtp._weights[labval];
    pos = m.find_next(pos);
  } // while
  return static_cast<label>(std::max_element(call.begin(), call.end()) - call.begin());
}

inline std::tuple<bool, std::size_t>
pure_enough(const DataMatrix& dm, const Monitor& child, const DecisionTreeParameters& dtp) {
  auto& labels = column(dm, 0);
  auto pos = child.find_first();
  std::array<dtype, 2> counts = { 0, 0 }; // how many of each label?

  while ( pos != Monitor::npos ) {
    counts[labels[pos]]++;
    pos = child.find_next(pos);
  } // while

  std::size_t total = 0;
  for ( std::size_t i = 0; i < counts.size(); ++i ) {
    if ( counts[i] < dtp._minClassSamples )
      return std::make_tuple(true, counts[0]*dtp._weights[0]>counts[1]*dtp._weights[1] ? 0 : 1);
    total += counts[i];
  } // for

  if ( total < dtp._minLeafSamples )
    return std::make_tuple(true, counts[0]*dtp._weights[0]>counts[1]*dtp._weights[1] ? 0 : 1);

  return std::make_tuple(std::get<1>(purity(counts, dtp._weights)) >= dtp._minPuritySplit, counts[0]*dtp._weights[0]>counts[1]*dtp._weights[1] ? 0 : 1);
}

inline std::tuple<dtype, std::array<dtype, 2>>
spit_purity(const DataMatrix& dm, const Monitor& child, const DecisionTreeParameters& dtp) {
  auto& labels = column(dm, 0);
  auto pos = child.find_first();
  std::array<dtype, 2> counts = { 0, 0 }; // how many of each label?

  while ( pos != Monitor::npos ) {
    counts[labels[pos]]++;
    pos = child.find_next(pos);
  } // while
  return std::make_tuple(std::get<1>(purity(counts, dtp._weights)), counts);
}

/*
  DecisionTreeClassifier
*/
struct DecisionTreeClassifier {
  explicit DecisionTreeClassifier(DecisionTreeParameters& dtp)
    : _dtp(dtp) {}

  void
  classify(const DataMatrixInv& newData, std::vector<label>& predictions) const {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before applying one!");

    std::size_t sz = newData.size1();
    for ( std::size_t i = 0; i < sz; ++i )
      predictions[i] = classify(row(newData, i));
  }

  label
  classify(const RowVector& row) const {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before you can use one!");

    auto currCore = std::get<PARENT>(_nodes.front());
    for ( auto iter = _nodes.begin(); iter != _nodes.end(); ) {
      auto& parent = *iter++;

      if ( std::get<PARENT>(parent) != currCore )
        continue;

      dtype absdiff = std::abs(std::get<THOLD>(*currCore) - row[std::get<FID>(*currCore)]);
      bool equal = (absdiff <= std::numeric_limits<dtype>::epsilon());
      if ( equal || std::get<THOLD>(*currCore) > row[std::get<FID>(*currCore)]  ) { // look left
        if ( !std::get<ISLEAF>(*currCore) )
          currCore = std::get<LEFTCHILD>(parent);
        else
          return std::get<DECISION>(*currCore);
      } else { // look right
        if ( !std::get<ISLEAF>(*currCore) )
          currCore = std::get<RIGHTCHILD>(parent);
        else
          return std::get<DECISION>(*currCore);
      }
    } // for
    throw std::logic_error("Problem with Tree!  See classify()");
  }

private:
  static constexpr std::size_t QS = 0;
  static constexpr std::size_t TH = 1;
  static constexpr std::size_t LP = 2;
  static constexpr std::size_t RP = 3;

  // Node's Core pointer elements
  static constexpr std::size_t PARENT = 0;
  static constexpr std::size_t LEFTCHILD = 1;
  static constexpr std::size_t RIGHTCHILD = 2;

  // Core's elements
  static constexpr std::size_t FID = 0;
  static constexpr std::size_t THOLD = 1;
  static constexpr std::size_t ISLEAF = 2;
  static constexpr std::size_t DECISION = 3;

  typedef featureID FeatureID;
  typedef dtype SplitValue;
  typedef bool IsLeaf;
  typedef std::size_t Decision;
  typedef std::tuple<FeatureID, SplitValue, IsLeaf, Decision> Core;
  typedef Core LeftChild;
  typedef LeftChild RightChild;
  typedef std::tuple<Core*, LeftChild*, RightChild*> Node;

protected:
  void
  split(const DataMatrix& dm, std::size_t nSplitFeatures) {
    /* main algorithm for learning splits, implemented as BFS */
    const std::size_t numSamples = dm.size1();
    Monitor parentMonitor(numSamples);
    parentMonitor.set();

    // find_split() returns feature ID, split value, a Monitor showing which bits
    //  'go to the right', left child decision&purity and right child decision&purity
    static constexpr std::size_t FID = 0;
    static constexpr std::size_t SPV = 1;
    static constexpr std::size_t MON = 2;
    static constexpr std::size_t LPU = 3;
    static constexpr std::size_t RPU = 4;

    // for sub-tuple of LPU or RPU
    static constexpr std::size_t DEC = 0;
    static constexpr std::size_t PUR = 1;

    const bool noLeaf = false;
    const bool isLeaf = true;
    const std::size_t noDecision = std::numeric_limits<std::size_t>::max();
    auto p = find_split(dm, nSplitFeatures, parentMonitor);
    Node current(new Core(std::get<FID>(p), std::get<SPV>(p), noLeaf, noDecision), nullptr, nullptr); // root
    decltype(p) q;

    // partition parent into left/right
    Monitor rightChildMonitor = std::get<MON>(p);
    Monitor leftChildMonitor = ~rightChildMonitor & parentMonitor;

    typedef std::tuple<Core*, Monitor, Monitor, std::size_t> internal;
    static constexpr std::size_t PARENT_CPTR = 0;
    static constexpr std::size_t LC_CPTR = 1, LC_MON = LC_CPTR;
    static constexpr std::size_t RC_CPTR = 2, RC_MON = RC_CPTR;
    static constexpr std::size_t LEVEL = 3;

    if ( done(std::get<PUR>(std::get<LPU>(p)), std::get<PUR>(std::get<RPU>(p))) ) {
      std::get<LC_CPTR>(current) = new Core(std::get<FID>(p), std::get<SPV>(p), isLeaf, noDecision);
      std::get<FID>(*std::get<LC_CPTR>(current)) = 0;
      std::get<SPV>(*std::get<LC_CPTR>(current)) = 0;
      std::get<DECISION>(*std::get<LC_CPTR>(current)) = std::get<DEC>(std::get<LPU>(p));
      std::get<RC_CPTR>(current) = new Core(std::get<FID>(p), std::get<SPV>(p), isLeaf, noDecision);
      std::get<FID>(*std::get<RC_CPTR>(current)) = 0;
      std::get<SPV>(*std::get<RC_CPTR>(current)) = 0;
      std::get<DECISION>(*std::get<RC_CPTR>(current)) = std::get<DEC>(std::get<RPU>(p));


      std::cout << "[" << std::endl;
      auto& curr = *std::get<PARENT_CPTR>(current);
      bool isleaf = std::get<ISLEAF>(curr);
      std::cout << std::get<FID>(curr) << " " << std::get<SPV>(curr) << " "
                << isleaf << " " << (isleaf ? std::get<DECISION>(curr) : 0) << std::endl;

      auto& durr = *std::get<LC_CPTR>(current);
      isleaf = std::get<ISLEAF>(durr);
      std::cout << std::get<FID>(durr) << " " << std::get<SPV>(durr) << " "
                << isleaf << " " << (isleaf ? std::get<DECISION>(durr) : 0) << std::endl;

      auto& eurr = *std::get<RC_CPTR>(current);
      isleaf = std::get<ISLEAF>(eurr);
      std::cout << std::get<FID>(eurr) << " " << std::get<SPV>(eurr) << " "
                << isleaf << " " << (isleaf ? std::get<DECISION>(eurr) : 0) << std::endl;

      std::cout << "]" << std::endl;
      return;
    }


    std::cout << "[" << std::endl;

    internal root_internal = std::make_tuple(std::get<PARENT_CPTR>(current), leftChildMonitor, rightChildMonitor, 0);
    std::queue<internal> que;
    que.push(root_internal);

    // breadth first traversal
    std::size_t nleaves = 0;
    while ( !que.empty() ) {
      auto& e = que.front();
      const std::size_t level = std::get<LEVEL>(e);
      current = Node(std::get<PARENT_CPTR>(e), nullptr, nullptr);
      bool leftIn = false;
      if ( std::get<LC_MON>(e).any() ) {
        leftIn = true;
        auto check = pure_enough(dm, std::get<LC_MON>(e), _dtp);
        if ( std::get<0>(check) ) { // pure enough to be a leaf
          const std::size_t decision = std::get<1>(check);
          std::get<LC_CPTR>(current) = new Core(0, 0, isLeaf, decision);
          Monitor unmonitored;
          auto nextl = std::make_tuple(std::get<LC_CPTR>(current), unmonitored, unmonitored, level+1);
          que.push(nextl);
          ++nleaves;
        } else {
          p = find_split(dm, nSplitFeatures, std::get<LC_MON>(e));
          std::get<LC_CPTR>(current) = new Core(std::get<FID>(p), std::get<SPV>(p), noLeaf, noDecision);
          parentMonitor = std::get<LC_MON>(e);
          rightChildMonitor = std::get<MON>(p);
          leftChildMonitor = ~rightChildMonitor & parentMonitor;

          if ( !rightChildMonitor.any() || rightChildMonitor == parentMonitor ) { // did not split, or all split left
            p = man_split(dm, parentMonitor); // force a split
            std::get<FID>(*std::get<LC_CPTR>(current)) = std::get<FID>(p);
            std::get<SPV>(*std::get<LC_CPTR>(current)) = std::get<SPV>(p);
            parentMonitor = std::get<LC_MON>(e);
            rightChildMonitor = std::get<MON>(p);
            leftChildMonitor = ~rightChildMonitor & parentMonitor;
            if ( rightChildMonitor == parentMonitor ) { // cannot be split further
              std::get<FID>(*std::get<LC_CPTR>(current)) = 0;
              std::get<SPV>(*std::get<LC_CPTR>(current)) = 0;
              std::get<ISLEAF>(*std::get<LC_CPTR>(current)) = true;
              std::get<DECISION>(*std::get<LC_CPTR>(current)) = weighted_vote_majority(dm, parentMonitor, _dtp);
              ++nleaves;
              leftChildMonitor.reset();
              rightChildMonitor.reset();
            }
          }
          auto nextl = std::make_tuple(std::get<LC_CPTR>(current), leftChildMonitor, rightChildMonitor, level+1);
          que.push(nextl);
        }
      }

      if ( std::get<RC_MON>(e).any() ) {
        if ( !leftIn ) { // must keep sibling that comes first
          std::get<LC_CPTR>(current) = new Core(0, 0, 1, 0); // leaf with decision=0
          Monitor unmonitor;
          auto nextl = std::make_tuple(std::get<LC_CPTR>(current), unmonitor, unmonitor, level+1);
          que.push(nextl);
        }
        auto check = pure_enough(dm, std::get<RC_MON>(e), _dtp);
        if ( std::get<0>(check) ) { // pure enough to be a leaf
          const std::size_t decision = std::get<1>(check);
          std::get<RC_CPTR>(current) = new Core(0, 0, isLeaf, decision);
          Monitor unmonitored;
          auto nextr = std::make_tuple(std::get<RC_CPTR>(current), unmonitored, unmonitored, level+1);
          que.push(nextr);
          ++nleaves;
        } else {
          q = find_split(dm, nSplitFeatures, std::get<RC_MON>(e));
          std::get<RC_CPTR>(current) = new Core(std::get<FID>(q), std::get<SPV>(q), noLeaf, noDecision);
          parentMonitor = std::get<RC_MON>(e);
          rightChildMonitor = std::get<MON>(q);
          leftChildMonitor = ~rightChildMonitor & parentMonitor;
          if ( !rightChildMonitor.any() || rightChildMonitor == parentMonitor ) { // did not split, or all split left
            q = man_split(dm, parentMonitor); // force a split
            std::get<FID>(*std::get<RC_CPTR>(current)) = std::get<FID>(q);
            std::get<SPV>(*std::get<RC_CPTR>(current)) = std::get<SPV>(q);
            parentMonitor = std::get<RC_MON>(e);
            rightChildMonitor = std::get<MON>(q);
            leftChildMonitor = ~rightChildMonitor & parentMonitor;
            if ( rightChildMonitor == parentMonitor ) { // cannot be split further
              std::get<FID>(*std::get<RC_CPTR>(current)) = 0;
              std::get<SPV>(*std::get<RC_CPTR>(current)) = 0;
              std::get<ISLEAF>(*std::get<RC_CPTR>(current)) = true;
              std::get<DECISION>(*std::get<RC_CPTR>(current)) = weighted_vote_majority(dm, parentMonitor, _dtp);
              ++nleaves;
              leftChildMonitor.reset();
              rightChildMonitor.reset();
            }
          }
          auto nextr = std::make_tuple(std::get<RC_CPTR>(current), leftChildMonitor, rightChildMonitor, level+1);
          que.push(nextr);
        }
      } else if ( leftIn ) { // make symmetric with null
        std::get<RC_CPTR>(current) = new Core(0, 0, 1, 0); // leaf with decision=0
        Monitor unmonitor;
        auto nextr = std::make_tuple(std::get<RC_CPTR>(current), unmonitor, unmonitor, level+1);
        que.push(nextr);
      }

      auto& curr = *std::get<PARENT_CPTR>(current);
      bool isleaf = std::get<ISLEAF>(curr);
      std::cout << std::get<FID>(curr) << " " << std::get<SPV>(curr) << " "
                << isleaf << " " << (isleaf ? std::get<DECISION>(curr) : 0) << std::endl;

      delete std::get<PARENT_CPTR>(current);
      que.pop();
    } // while

    std::cout << "]" << std::endl;

  }

  // may add in _minLeafSamples eventually
  inline bool
  done(dtype lchildPurity, dtype rchildPurity, std::size_t level, std::size_t nLeaves) {
    return ((lchildPurity >= _dtp._minPuritySplit) && (rchildPurity >= _dtp._minPuritySplit)) ||
           (_dtp._maxDepth && level > _dtp._maxDepth) ||
           (_dtp._maxLeafNodes && nLeaves >= _dtp._maxLeafNodes);
  }

  inline bool
  done(dtype lchildPurity, dtype rchildPurity) {
    return (lchildPurity >= _dtp._minPuritySplit) && (rchildPurity >= _dtp._minPuritySplit);
  }

public:
  ~DecisionTreeClassifier() {
    auto jiter = _nodes.begin();
    if ( jiter != _nodes.end() ) {
      for ( auto iter = _nodes.begin(); iter != _nodes.end(); ) {
        if ( std::get<PARENT>(*iter) )
          delete std::get<PARENT>(*iter);
        if ( ++iter != _nodes.end() )
          ++jiter;
      } // for

      if ( std::get<LEFTCHILD>(*jiter) )
        delete std::get<LEFTCHILD>(*jiter);
      if ( std::get<RIGHTCHILD>(*jiter) )
        delete std::get<RIGHTCHILD>(*jiter);
    }
  }

  void
  learn(const DataMatrix& dm) {
    const std::size_t numFeatures = dm.size2();
    std::size_t nSplitFeatures = 0;
    switch(_dtp._splitMaxSelect) {
      case SplitMaxFeat::INT:
        nSplitFeatures = static_cast<std::size_t>(_dtp._selectSplitNumeric);
        break;
      case SplitMaxFeat::FLT:
        nSplitFeatures = static_cast<std::size_t>(std::round(_dtp._selectSplitNumeric * numFeatures));
        break;
      case SplitMaxFeat::SQRT:
        nSplitFeatures = static_cast<std::size_t>(std::round(std::sqrt(numFeatures)));
        break;
      case SplitMaxFeat::LOG2:
        nSplitFeatures = static_cast<std::size_t>(std::round(std::log2(numFeatures)));
        break;
      default:
        throw std::domain_error("program error: unknown split type in learn()");
    }
    if ( nSplitFeatures <= 1 )
      throw std::domain_error("Unable to fulfill requested nFeatures to split upon (value <= 1).  Did you mean a float like 1.0?");
    else if ( nSplitFeatures > numFeatures )
      throw std::domain_error("Unable to fulfill requested nFeatures to split upon (value > #columns).");

    split(dm, nSplitFeatures);
  }

  friend inline // read in a model previously created
  std::istream&
  operator>>(std::istream& input, DecisionTreeClassifier& dtc) {
    DecisionTreeClassifier::FeatureID fid; // feature id?
    DecisionTreeClassifier::SplitValue spv; // split value?
    bool ilf; // is leaf?
    std::size_t dsn; // decision?
    std::string open_bracket, closed_bracket;

    input >> open_bracket;
    if ( open_bracket.empty() )
      return input; // so annoying
    std::queue<Core*> que;
    input >> fid;
    input >> spv;
    input >> ilf;
    input >> dsn;
    Core* parent = new Core(fid, spv, ilf, dsn);
    que.push(parent);
    while ( !que.empty() ) {
      parent = que.front();
      Core* left = nullptr;
      Core* right = nullptr;
      if ( !std::get<DecisionTreeClassifier::ISLEAF>(*parent) ) { // not a leaf
        input >> fid;
        input >> spv;
        input >> ilf;
        input >> dsn;
        left = new Core(fid, spv, ilf, dsn);

        input >> fid;
        input >> spv;
        input >> ilf;
        input >> dsn;
        right = new Core(fid, spv, ilf, dsn);

        que.push(left);
        que.push(right);
      }
      dtc._nodes.push_back(Node(parent, left, right));
      que.pop();
    } // while

    input >> closed_bracket;
    return input;
  }

  friend inline // write out this model
  std::ostream&
  operator<<(std::ostream& output, const DecisionTreeClassifier& dtc) {
    static constexpr std::size_t PAR = dtc.PARENT;

    static constexpr std::size_t FID = dtc.FID;
    static constexpr std::size_t SPV = dtc.THOLD;
    static constexpr std::size_t ILF = dtc.ISLEAF;
    static constexpr std::size_t DEC = dtc.DECISION;

    output << "[" << std::endl;

    const std::size_t na = 0;
    for ( auto& n : dtc._nodes ) {
      if ( std::get<PAR>(n) ) {
        const Core& parent = *std::get<PAR>(n);
        bool isleaf = std::get<ILF>(parent);
        output << std::get<FID>(parent) << " " << std::get<SPV>(parent) << " "
               << isleaf << " " << (isleaf ? std::get<DEC>(parent) : na) << std::endl;
      }
    } // for

    output << "]";
    return output;
  }

protected:

  template <typename NonZeroes>
  std::pair<dtype, dtype>
  get_range(const NonZeroes& cont, bool includeZeroes = false) {
    auto first = cont.begin(), last = cont.end();
    if ( first == last )
      return std::make_pair(0,0);
    auto mx = cont.begin(), mn = cont.begin();
    std::size_t cntr = 0;
    while (++first!=last) { // looping over non-zero elements only
      if ( *mx<*first ) mx=first;
      if ( *mn>*first ) mn=first;
      ++cntr;
    } // while
    if ( includeZeroes ) {
      if ( *mn>0 )
        return std::make_pair(0, *mx);
    }
    return std::make_pair(*mn, *mx);
  }

  template <typename C>
  inline dtype
  gini(const C& c, std::size_t totalSz) {
    if ( 0 == totalSz ) { return 0; }
    dtype sqsum = 0;
    for ( std::size_t cls = 0; cls < c.size(); ++cls )
      sqsum += std::pow(c[cls]/totalSz, 2.0);
    return 1-sqsum;
  }

  template <typename C>
  inline dtype
  gini(const C& left, const C& right) {
    const dtype nleft = std::accumulate(left.begin(), left.end(), 0);
    const dtype nright = std::accumulate(right.begin(), right.end(), 0);
    return (nleft/(nleft+nright))*gini(left, nleft) + (nright/(nright+nleft))*gini(right, nright);
  }

  //===================================
  // return [0] as quality of split
  // return [1] is threshold chosen
  // return [2] is decision&purity in left branch
  // return [3] is decision&purity in right branch
  //  for now, assume 2 classes/labels of 0 and 1 as we index arrays of size 2.
  //  use const iterators or for-range if not considering implicit (0) feature values
  //    use const: 5.2 at
  //    http://www.boost.org/doc/libs/1_51_0/libs/numeric/ublas/doc/operations_overview.htm#4SubmatricesSubvectors
  std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>>
  split_gini(const DataMatrix& dm, std::size_t featColumn, SplitStrategy s, bool includeZeroes, const Monitor& monitor) {
    const auto& labels = ub::column(dm, 0);
    const auto& values = ub::column(dm, featColumn);
    auto iter = values.begin(), iter_end = values.end();
    std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>> rtn(0,0,std::make_tuple(0,0), std::make_tuple(0,0));
    Monitor mask(monitor);
    if ( s == SplitStrategy::BEST ) { // find best value to split on; smallest gini is best
      std::set<dtype> uniq_scores;
      while ( iter != iter_end ) { // const iterators
        if ( monitor[iter.index()] )
          uniq_scores.insert(*iter);
        mask[iter.index()] = false;
        ++iter;
      } // while
      if ( includeZeroes && mask.any() )
        uniq_scores.insert(0);
 
      dtype best_score = std::numeric_limits<dtype>::max();
      for ( auto u : uniq_scores ) {
        auto v = u + std::numeric_limits<dtype>::epsilon();
        std::array<dtype, 2> nl = { 0, 0 }; // how many instances of each class in left branch
        auto nr = nl; //  how many instances of each class in right branch
        if ( includeZeroes ) {
          for ( std::size_t i = 0; i < values.size(); ++i ) {
            if ( monitor[i] ) {
              if ( values[i] > v ) // v not u
                nr[labels[i]]++;
              else
                nl[labels[i]]++;
            }
          } // for
        } else { // only look at labels of rows with nonZero values for this featColumn
          for ( auto viter = values.begin(); viter != values.end(); ++viter ) { // these are const_iterators due to values
            if ( monitor[viter.index()] ) {
              if ( *viter > v ) // v, not u
                nr[labels[viter.index()]]++;
              else
                nl[labels[viter.index()]]++;
            }
          } // for
        }
        dtype gval = gini(nl, nr);
        if ( gval < best_score ) {
          std::get<QS>(rtn) = gval;
          std::get<TH>(rtn) = u; // u, not v
          std::get<LP>(rtn) = purity(nl, _dtp._weights);
          std::get<RP>(rtn) = purity(nr, _dtp._weights);
          best_score = gval;
          if ( best_score == 0 )
            break;
        }
      } // for
    } else if ( s == SplitStrategy::RANDOM ) {
      auto range = get_range(values, includeZeroes);
      double crn = _dtp._unif(_dtp._rng);
      std::get<TH>(rtn) = range.first + crn*(range.second-range.first);
      std::array<dtype, 2> nl = { 0, 0 }; // how many instances of each class in left branch
      auto nr = nl; //  how many instances of each class in right branch
      if ( includeZeroes ) {
        for ( std::size_t i = 0; i < values.size(); ++i ) {
          if ( values[i] > std::get<TH>(rtn) )
            nr[labels(i)]++;
          else
            nl[labels(i)]++;
        } // for
      } else { // only look at labels of rows with nonZero values for this featColumn
        for ( auto viter = values.begin(); viter != values.end(); ++viter ) { // these are const iterators due to values
          if ( *viter > std::get<TH>(rtn) )
            nr[labels(viter.index())]++;
          else
            nl[labels(viter.index())]++;
        } // for
      }
      std::get<QS>(rtn) = gini(nl, nr);
      std::get<LP>(rtn) = purity(nl, _dtp._weights);
      std::get<RP>(rtn) = purity(nr, _dtp._weights);
    } else {
      throw std::domain_error("unknown split strategy");
    }
    return rtn;
  }

  inline
  /* quality, split-value, decision-left/purity-left, decision-right/purity-right */
  std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>>
  evaluate(const DataMatrix& dm, std::size_t featColumn, const Monitor& monitor) {
    switch(_dtp._criterion) {
      case Criterion::GINI:
        return split_gini(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision, monitor);
      case Criterion::ENTROPY:
return std::make_tuple(0,0,std::make_tuple(0,0),std::make_tuple(0,0));
//        return split_entropy(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision, monitor);
      default:
          throw std::domain_error("Unknown criterion in evaluate()");
    }
  }

  std::tuple<featureID, dtype, Monitor, std::tuple<label, dtype>, std::tuple<label, dtype>>
  man_split(const DataMatrix& dm, Monitor mask) {
    /* this is an expensive function which should not be called except in extreme cases as a final save to
         split more rows.  if this is being called fairly often, you need to increase the number of splits
         at each step or use zeroes or ...
       this is a greedy, first come, first served function that does not attempt to find any optimal split
         value - it will use the first that it can find that separates two monitored rows with differing
         labels, if possible.
    */
    const auto& labels = ub::column(dm, 0);
    const label unset = std::numeric_limits<label>::max();
    const bool done = false;
    std::size_t a = 0, b = 0;
    std::size_t idx = 0;
    bool first = true;
    while ( !done ) {
      // find two monitored rows with different labels
      if ( first )
        idx = mask.find_first();
      else
        idx = mask.find_next(a);

      label A = unset;
      while ( idx != Monitor::npos ) {
        if ( A != unset ) {
          if ( labels[idx] != A ) {
            b = idx;
            break;
          }
        } else {
          A = labels[idx];
          a = idx;
        }
        idx = mask.find_next(idx);
      } // while

      if ( idx == Monitor::npos ) { // ultimate stopping condition
        // all monitored rows from a->#rows are labeled the same; mask is unchanged
        return std::make_tuple(0, 0, mask, std::make_tuple(0,1), std::make_tuple(0,1));
      }

      // find first feature that can separate these different labels
      for ( std::size_t features = 1; features < dm.size2(); ++features ) {
        const auto& values = ub::column(dm, features);
        if ( !_dtp._useImpliedZerosInSplitDecision && (values[a]==0 || values[b]==0) )
          continue;
        dtype absdiff = std::abs(values[a]-values[b]);
        bool equal = (absdiff <= 2*std::numeric_limits<dtype>::epsilon()); // mult2 since div2 below
        if ( !equal ) { // at least these two rows can be separated
          const dtype two = 2.0;
          dtype value = (values[a]+values[b])/two;

          std::array<dtype, 2> nl = { 0, 0 }; // how many instances of each class in left branch
          auto nr = nl; //  how many instances of each class in right branch
          if ( _dtp._useImpliedZerosInSplitDecision ) {
            for ( std::size_t i = 0; i < values.size(); ++i ) {
              if ( mask[i] ) {
                if ( values[i] > value )
                  nr[labels[i]]++;
                else {
                  nl[labels[i]]++;
                  mask[i] = false;
                }
              }
            } // for
          } else { // only look at labels of rows with nonZero values for this featColumn
            for ( auto viter = values.begin(); viter != values.end(); ++viter ) { // these are const_iterators due to values
              if ( mask[viter.index()] ) {
                if ( *viter > value )
                  nr[labels[viter.index()]]++;
                else {
                  nl[labels[viter.index()]]++;
                  mask[viter.index()] = false;
                }
              }
            } // for
          }
          return std::make_tuple(features, value, mask, purity(nl, _dtp._weights), purity(nr, _dtp._weights));
        }
      } // for

      // found no feature to separate these different labels...try again?
      first = false;
    } // while
    throw std::domain_error("Not supposed to reach here: man_split()");
  }

  std::tuple<featureID, dtype, Monitor, std::tuple<label, dtype>, std::tuple<label, dtype>>
  get_best(const DataMatrix& dm, Monitor mask) {
    dtype currBest = -1;
    dtype currValue = 0;
    std::tuple<label, dtype> currLPurity(0,0);
    std::tuple<label, dtype> currRPurity(0,0);
    featureID currCol = 0;
    const std::size_t nfeats = dm.size2();
    for ( std::size_t nxt = 1; nxt < nfeats; ++nxt ) {
      auto val = evaluate(dm, nxt, mask); // return split quality, split value, ldec/lpurity, rdec/rpurity
      if ( std::get<QS>(val) > currBest ) {
        currBest = std::get<QS>(val);
        currValue = std::get<TH>(val);
        currLPurity = std::get<LP>(val);
        currRPurity = std::get<RP>(val);
        currCol = nxt;
        if ( currBest == 1 )
          break;
      }
    } // for

    if ( currCol == 0 ) //can happen if allZeroes() with the given mask -> punt...
      return std::make_tuple(1, 0, mask, std::make_tuple(0,0), std::make_tuple(0,0));

    const auto& vals = column(dm, currCol);
    const dtype threshold = currValue;
    std::size_t beg = mask.find_first();
    while ( beg != mask.npos ) {
      if ( threshold >= vals[beg] ) // goes to left child (could be implicit zero)
        mask[beg] = false;
      beg = mask.find_next(beg);
    } // while
    return std::make_tuple(currCol, threshold, mask, currLPurity, currRPurity);
  }

  // find_split() returns feature ID, split value and monitor showing bit masking ->
  //    consider only those set
  //  If stays set on output, split row 'to the right', otherwise it goes to the left
  //  Also return the left and right purities of child nodes
  std::tuple<featureID, dtype, Monitor, std::tuple<label, dtype>, std::tuple<label, dtype>>
  find_split(const DataMatrix& dm, std::size_t nSplitFeatures, Monitor mask) {
    if ( _dtp._selectSplitNumeric == 1.0 )
      return get_best(dm, mask);
    const std::size_t numFeatures = dm.size2() - 1; // includes label column
    std::size_t featureCount = 0;
    dtype currBest = -1;
    dtype currValue = 0;
    std::tuple<label, dtype> currLPurity(0,0);
    std::tuple<label, dtype> currRPurity(0,0);
    featureID currCol = 0;
    bool done = false;
    Monitor featMask(dm.size2()); // no -1 here
    featMask.set();
    featMask[0] = false; // label
    std::size_t lst = 0;
    while ( featureCount < nSplitFeatures ) {
      featureID col = 0;
      std::size_t save_me = 0;
      static const std::size_t loopy = 50;
      while (true) {
        col = static_cast<featureID>(std::round(_dtp._unif(_dtp._rng) * numFeatures));
        if ( featMask[col] ) {
          featMask[col] = false;
          if ( !allZeroes(dm, col, mask) )
            break;
        }
        if ( ++save_me > loopy ) {
          std::size_t nxt = featMask.find_next(lst);
          if ( nxt != featMask.npos ) {
            featMask[col] = false;
            lst = nxt;
            if ( !allZeroes(dm, col, mask) )
              break;
          } else {
            done = true;
            break;
          }
        }
      } // while

      if ( done )
        break;

      auto val = evaluate(dm, col, mask); // return split quality, split value, ldec/lpurity, rdec/rpurity
      if ( std::get<QS>(val) > currBest ) {
        currBest = std::get<QS>(val);
        currValue = std::get<TH>(val);
        currLPurity = std::get<LP>(val);
        currRPurity = std::get<RP>(val);
        currCol = col;
        if ( currBest == 1 )
          break;
      }
      ++featureCount;
    } // while

    if ( done )
      return std::make_tuple(1, 0, mask, std::make_tuple(0,0), std::make_tuple(0,0));

    std::size_t beg = mask.find_first();
    if ( currCol == 0 || beg == mask.npos ) // can happen if allZeroes() with the given mask -> punt...
      return std::make_tuple(1, 0, mask, std::make_tuple(0,0), std::make_tuple(0,0));
//should also return the featMask object so that calling routine can pass it back in as a starting point
    const auto& vals = column(dm, currCol);
    const dtype threshold = currValue;
    while ( beg != mask.npos ) {
      if ( threshold >= vals[beg] ) // goes to left child (could be implicit zero)
        mask[beg] = false;
      beg = mask.find_next(beg);
    } // while
    return std::make_tuple(currCol, threshold, mask, currLPurity, currRPurity);
  }

public: // sjn, for now
  DecisionTreeParameters& _dtp;
  std::list<Node> _nodes; // root is first element
};

} // namespace Tree

#endif // __ALTIUS_DTREE_HPP__
