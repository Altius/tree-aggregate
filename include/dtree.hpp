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
      _argmap[Utils::uppercase("MinPuritySplit")] = Utils::Nums::PLUSREAL_NOZERO; // split if purity less than this
      _argmap[Utils::uppercase("UseZeroesInSplit")] = Utils::Nums::BOOL; // use missing data == zero ?
      _argmap[Utils::uppercase("Seed")] = Utils::Nums::PLUSINT; // zero means use random seed
      _argmap[Utils::uppercase("SelectSplitNumeric")] = Utils::Nums::PLUSREAL; // if SplitMaxFeatures=INT||FLT, this is the value
    }

  friend
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
    output << space << "MinPuritySplit=" << dtp._minPuritySplit;
    output << space << "UseZeroesInSplit=" << dtp._useImpliedZerosInSplitDecision;
    output << space << "ClassWeightType=" << int(dtp._classWeightType);
    output << space << "Seed=" << dtp._seed;
    output << space << "SelectSplitNumeric=" << dtp._selectSplitNumeric;
    return output;
  }

  friend
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
        else if ( jev[0] == Utils::uppercase("MinPuritySplit") )
          dtp.set_value(dtp._minPuritySplit, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("UseZeroesInSplit") )
          dtp.set_value(dtp._useImpliedZerosInSplitDecision, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("Seed") )
          dtp.set_value(dtp._seed, jev[0], jev[1]);
        else if ( jev[0] == Utils::uppercase("SelectSplitNumeric") )
          dtp.set_value(dtp._selectSplitNumeric, jev[0], jev[1]);
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

private:
  friend struct DecisionTreeClassifier;

  // by construction (would be const if not for operator>> and my laziness to do it differently)
  Criterion _criterion;
  SplitStrategy _splitStrategy;
  SplitMaxFeat _splitMaxSelect; // if INT, then _selectSplitNumeric > 0.  if FLT, then 0 < _selectSplitNumeric <= 1
  std::size_t _maxDepth; // 0 means expand until all leaves meet purity criterion or until all nodes contain less than _minLeafSamples
  std::size_t _maxLeafNodes; // 0, then unlimited leaf nodes, else grow a tree until you reach _maxLeafNodes using bfs
  std::size_t _minLeafSamples; // 0, then not a stopping condition, else done splitting if #samples in leaf drops to this level
  dtype _minPuritySplit; // A node will split if its purity is less than the threshold, otherwise it is a leaf
  bool _useImpliedZerosInSplitDecision;
  ClassWeightType _classWeightType;
  long int _seed;

  // set after or while data are read
  dtype _selectSplitNumeric; // if _splitMaxSelect is INT,FLT then this is the value of the INT or FLT

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
purity(const std::array<dtype, 2>& a) {
  return std::make_pair(a[0]>a[1] ? 0 : 1, std::max(a[0], a[1])/(a[0]+a[1]));
}

inline label
majority_decision(const std::array<dtype, 2>& a) {
  return a[0] > a[1] ? 0 : 1;
}

/*
  DecisionTreeClassifier
*/
struct DecisionTreeClassifier {
  explicit DecisionTreeClassifier(DecisionTreeParameters& dtp)
    : _dtp(dtp) {}

  std::list<label>
  classify(const DataMatrixInv& newData) const {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before applying one!");

    std::list<label> toRtn;
    std::size_t sz = newData.size1();
    for ( std::size_t i = 0; i < sz; ++i )
      toRtn.push_back(classify(row(newData, i)));
    return toRtn;
  }

  label
  classify(const RowVector& row) const {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before you can use one!");

    auto currCore = std::get<PARENT>(_nodes.back());
    for ( auto iter = _nodes.rbegin(); iter != _nodes.rend(); ) {
      auto& parent = *iter++;

      if ( std::get<PARENT>(parent) != currCore )
        continue;

      if ( std::get<THOLD>(*currCore) > row[std::get<FID>(*currCore)]  ) { // look left
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
    const std::size_t numFeatures = dm.size2();
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

// sjn might make better sense to make the 3 monitors pointers everywhere; test performance later
    const bool noLeaf = false;
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

    std::queue<internal> que;
    que.push(std::make_tuple(std::get<PARENT_CPTR>(current), leftChildMonitor, rightChildMonitor, 0));

    // breadth first traversal
    std::size_t nleaves = 0;
    while ( !que.empty() ) {
      auto& e = que.front();
      const std::size_t level = std::get<LEVEL>(e);

      current = Node(std::get<PARENT_CPTR>(e), nullptr, nullptr);
      if ( std::get<LC_MON>(e).any() ) {
        p = find_split(dm, nSplitFeatures, std::get<LC_MON>(e));
        std::get<LC_CPTR>(current) = new Core(std::get<FID>(p), std::get<SPV>(p), noLeaf, noDecision);

        parentMonitor = std::get<LC_MON>(e);
        rightChildMonitor = std::get<MON>(p);
        leftChildMonitor = ~rightChildMonitor & parentMonitor;
        auto nextl = std::make_tuple(std::get<LC_CPTR>(current), leftChildMonitor, rightChildMonitor, level+1);

        if ( !done(std::get<PUR>(std::get<LPU>(p)), level, nleaves) )
          que.push(nextl);
        else {
          std::get<ISLEAF>(*std::get<LC_CPTR>(current)) = true;
          std::get<DECISION>(*std::get<LC_CPTR>(current)) = std::get<DEC>(std::get<LPU>(p));
          ++nleaves;
        }
      }

      if ( std::get<RC_MON>(e).any() ) {
        q = find_split(dm, nSplitFeatures, std::get<RC_MON>(e));
        std::get<RC_CPTR>(current) = new Core(std::get<FID>(q), std::get<SPV>(q), noLeaf, noDecision);

        parentMonitor = std::get<RC_MON>(e);
        rightChildMonitor = std::get<MON>(q);
        leftChildMonitor = ~rightChildMonitor & parentMonitor;
        auto nextr = std::make_tuple(std::get<RC_CPTR>(current), leftChildMonitor, rightChildMonitor, level+1);

        if ( !done(std::get<PUR>(std::get<RPU>(q)), level, nleaves) )
          que.push(nextr);
        else {
          std::get<ISLEAF>(*std::get<RC_CPTR>(current)) = true;
          std::get<DECISION>(*std::get<RC_CPTR>(current)) = std::get<DEC>(std::get<RPU>(p));
          ++nleaves;
        }
      }

      _nodes.push_back(current);
      que.pop();
    } // while
  }

  // may add in _minLeafSamples eventually
  inline bool
  done(dtype currPurity, std::size_t level, std::size_t maxLeaves) {
    return (currPurity >= _dtp._minPuritySplit) ||
           (_dtp._maxDepth && level > _dtp._maxDepth)   ||
           (_dtp._maxLeafNodes && maxLeaves > _dtp._maxLeafNodes);
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
    const std::size_t numSamples = dm.size1();

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
      throw std::domain_error("Unable to fulfill requested nFeatures to split upon (value <= 1).");
    else if ( nSplitFeatures > numFeatures )
      throw std::domain_error("Unable to fulfill requested nFeatures to split upon (value > #columns).");

    split(dm, nSplitFeatures);
  }

  friend // read in a model previously created
  std::istream&
  operator>>(std::istream& input, DecisionTreeClassifier& dtc) {
    if (!input)
      return input;

    DecisionTreeClassifier::FeatureID fid; // feature id?
    DecisionTreeClassifier::SplitValue spv; // split value?
    bool ilf; // is leaf?
    std::size_t dsn; // decision?
    std::string open_bracket, closed_bracket;

    std::queue<Core*> que;
input >> open_bracket;
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
input >> open_bracket;
    return input;
  }

  friend // write out this model
  std::ostream&
  operator<<(std::ostream& output, const DecisionTreeClassifier& dtc) {
    static constexpr std::size_t PAR = dtc.PARENT;
    static constexpr std::size_t LCHILD = dtc.LEFTCHILD;
    static constexpr std::size_t RCHILD = dtc.RIGHTCHILD;

    static constexpr std::size_t FID = dtc.FID;
    static constexpr std::size_t SPV = dtc.THOLD;
    static constexpr std::size_t ILF = dtc.ISLEAF;
    static constexpr std::size_t DEC = dtc.DECISION;

    output << "[" << std::endl;

    const std::size_t na = 0;
    for ( auto& n : dtc._nodes ) {
      const Core& parent = *std::get<PAR>(n);
      bool isleaf = std::get<ILF>(parent);
      output << std::get<FID>(parent) << " " << std::get<SPV>(parent) << " "
             << isleaf << " " << (isleaf ? std::get<DEC>(parent) : na) << std::endl;
    } // for

    Core const* lchild = std::get<LCHILD>(dtc._nodes.back());
    if ( lchild ) {
      bool isleaf = std::get<ILF>(*lchild);
      output << std::get<FID>(*lchild) << " " << std::get<SPV>(*lchild) << " "
             << isleaf << " " << (isleaf ? std::get<DEC>(*lchild) : na) << std::endl;
    }

    Core const* rchild = std::get<RCHILD>(dtc._nodes.back());
    if ( rchild ) {
      bool isleaf = std::get<ILF>(*rchild);
      output << std::get<FID>(*rchild) << " " << std::get<SPV>(*rchild) << " "
             << isleaf << " " << (isleaf ? std::get<DEC>(*rchild) : na) << std::endl;
    }

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
  dtype
  gini(const C& left, const C& right) {
    dtype class_total = 0, prop_left = 0, prop_right = 0;
    dtype nleft = std::accumulate(left.begin(), left.end(), 0);
    dtype nright = std::accumulate(right.begin(), right.end(), 0);
    dtype sum_left = 0, sum_right = 0;
    for ( std::size_t cls = 0; cls < left.size(); ++cls ) {
      class_total = left[cls] + right[cls];
      if ( 0 == class_total )
        continue;
      prop_left = left[cls]/class_total;
      prop_right = right[cls]/class_total;
      sum_left += (prop_left*(1-prop_left));
      sum_right += (prop_right*(1-prop_right));
    } // for
    return nleft*sum_left + nright*sum_right;
  }

//sjn need to keep track of whether we include zeroes in the output model of a tree for each applicable feature
// eventually make this a per-feature thing.  on first iteration, just use 1 bool to apply everywhere
  //===================================
  // return [0] as quality of split
  // return [1] is threshold chosen
  // return [2] is purity in left branch
  // return [3] is purity in right branch
  //  for now, assume 2 classes/labels of 0 and 1 as we index arrays of size 2.
  //  use const iterators or for-range if not considering implicit (0) feature values
  //    use const: 5.2 at
  //    http://www.boost.org/doc/libs/1_51_0/libs/numeric/ublas/doc/operations_overview.htm#4SubmatricesSubvectors
  std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>>
  split_gini(const DataMatrix& dm, std::size_t featColumn, SplitStrategy s, bool includeZeroes) {
    const auto& labels = ub::column(dm, 0);
    const auto& values = ub::column(dm, featColumn);
    auto iter = values.begin(), iter_end = values.end();
    std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>> rtn(0,0,std::make_tuple(0,0), std::make_tuple(0,0));
    if ( s == SplitStrategy::BEST ) { // find best value to split on; smallest gini is best
      std::set<dtype> uniq_scores(iter, iter_end);
      if ( includeZeroes )
        uniq_scores.insert(0);

      dtype best_score = 1e8;
      for ( auto u : uniq_scores ) {
        std::array<dtype, 2> nl = { 0, 0 }; // how many instances of each class in left branch
        auto nr = nl; //  how many instances of each class in right branch
        if ( includeZeroes ) {
          for ( std::size_t i = 0; i < values.size(); ++i ) {
            if ( values[i] > u )
              nr[labels(i)]++;
            else
              nl[labels(i)]++;
          } // for
        } else { // only look at labels of rows with nonZero values for this featColumn
          for ( auto viter = values.begin(); viter != values.end(); ++viter ) { // these are const_iterators due to values
            if ( *viter > u )
              nr[labels(viter.index())]++;
            else
              nl[labels(viter.index())]++;
          } // for
        }
        dtype gval = gini(nl, nr);
        if ( gval < best_score ) {
          std::get<QS>(rtn) = 1-gval;
          std::get<TH>(rtn) = u;
          std::get<LP>(rtn) = purity(nl);
          std::get<RP>(rtn) = purity(nr);
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
      std::get<QS>(rtn) = 1-gini(nl, nr);
      std::get<LP>(rtn) = purity(nl);
      std::get<RP>(rtn) = purity(nr);
    }
    return rtn;
  }

  inline
  /* quality, split-value, decision-left/purity-left, decision-right/purity-right */
  std::tuple<dtype, dtype, std::tuple<label, dtype>, std::tuple<label, dtype>>
  evaluate(const DataMatrix& dm, std::size_t featColumn) {
    switch(_dtp._criterion) {
      case Criterion::GINI:
        return split_gini(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision);
      case Criterion::ENTROPY:
return std::make_tuple(0,0,std::make_tuple(0,0),std::make_tuple(0,0));
//        return split_entropy(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision);
      default:
          throw std::domain_error("Unknown criterion in evaluate()");
    }

  }

  // find_split() returns feature ID, split value and monitor showing bit masking -> consider only those set
  //  If stays set on output, split row 'to the right', otherwise it goes to the left
  //  Also return the left and right purities of child nodes
  std::tuple<featureID, dtype, Monitor, std::tuple<label, dtype>, std::tuple<label, dtype>>
  find_split(const DataMatrix& dm, std::size_t nSplitFeatures, Monitor mask) {
    const std::size_t numFeatures = dm.size2() - 1; // includes label column
    std::size_t featureCount = 0;
    dtype currBest = -1;
    dtype currValue = 0;
    std::tuple<label, dtype> currLPurity(0,0);
    std::tuple<label, dtype> currRPurity(0,0);
    featureID currCol = 0;
    std::set<std::size_t> selectedFeats;
    while ( featureCount < nSplitFeatures ) {
      featureID col = 0;
      while (true) {
// sjn: could loop forever on edge cases - fix this
        col = static_cast<featureID>(std::round(_dtp._unif(_dtp._rng) * numFeatures));
        if ( col > 0 && selectedFeats.insert(col).second && !allZeroes(dm, col, mask) )
          break;
      } // while

      auto val = evaluate(dm, col); // return split quality, split value, ldec/lpurity, rdec/rpurity
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

    std::size_t beg = mask.find_first();
    if ( currCol == 0 || beg == mask.npos )
      throw std::domain_error("something very wrong: find_split()");

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
