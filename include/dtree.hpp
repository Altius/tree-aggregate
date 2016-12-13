#ifndef __ALTIUS_DTREE_HPP__
#define __ALTIUS_DTREE_HPP__

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <forward_list>
#include <iostream>
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

namespace Tree {

namespace ub = boost::numeric::ublas;

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

struct ByLine : public std::string {
  friend std::istream& operator>>(std::istream& is, ByLine& b) {
    std::getline(is, b);
    return(is);
  }
};

std::vector<std::string>
split(const std::string& s, const std::string& d) {
  std::vector<std::string> rtn;
  std::string::size_type pos1 = 0, pos2 = 0;
  do {
    pos2 = s.find(d, pos1);
    rtn.push_back(s.substr(pos1, pos2-pos1));
    pos1 = pos2 + 1;
  } while ( pos2 != std::string::npos ); // while
  return rtn;
}

std::string toupper(const std::string& s) {
  std::string rtn = s;
  for ( std::size_t i = 0; i < rtn.size(); ++i )
    rtn[i] = std::toupper(rtn[i]);
  return rtn;
}

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
                         const SplitType& splitValue = SplitType(),
                         int maxDepth = 0,
                         int maxLeafNodes = 0,
                         dtype minSamplesLeaf = 1,
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
                    _splitMaxFeatures(-1),
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

      _enums[toupper("Criterion")] = static_cast<int>(_criterion);
      _enums[toupper("SplitStrategy")] = static_cast<int>(_splitStrategy);
      _enums[toupper("SplitMaxFeatures")] = static_cast<int>(_splitMaxSelect);
      _enums[toupper("ClassWeightType")] = static_cast<int>(_classWeightType);

      _argmap[toupper("MaxDepth")] = Nums::PLUSINT; // zero means unused; grow full
      _argmap[toupper("MaxLeafNodes")] = Nums::PLUSINT; // zero means unused
      _argmap[toupper("MinLeafSamples")] = Nums::PLUSREAL; // when set, split until Node has this number or fewer; FLT means % of #rows
      _argmap[toupper("MinPuritySplit")] = Nums::PLUSREAL_NOZERO; // split if purity less than this
      _argmap[toupper("UseZeroesInSplit")] = Nums::BOOL; // use missing data == zero ?
      _argmap[toupper("Seed")] = Nums::PLUSINT; // zero means use random seed
      _argmap[toupper("SelectSplitNumeric")] = Nums::PLUSREAL; // if SplitMaxFeatures=INT||FLT, this is the value
    }

  friend
  std::ostream&
  operator<<(std::ostream& output, const DecisionTreeParameters& dtp) {
    constexpr char space = ' ';
    output << "Criterion=" << int(dtp._criterion);
    output << space << "SplitStrategy=" << int(dtp._splitStrategy);
    output << space << "SplitMaxFeat=" << int(dtp._splitMaxSelect);
    output << space << "MaxDepth=" << dtp._maxDepth;
    output << space << "MaxLeafNodes=" << dtp._maxLeafNodes;
    output << space << "MinLeafSamples=" << dtp._minLeafSamples;
    output << space << "MinPuritySplit=" << dtp._minPuritySplit;
    output << space << "UseZeroesInSplit=" << dtp._useImpliedZerosInSplitDecision;
    output << space << "ClassWeighting=" << int(dtp._classWeightType);
    output << space << "Seed=" << dtp._seed;
    output << space << "SplitMaxFeatures=" << int(dtp._splitMaxFeatures);
    output << space << "SelectSplitNumeric=" << dtp._selectSplitNumeric;
    output << std::endl;
    return output;
  }

  friend
  std::istream&
  operator>>(std::istream& is, DecisionTreeParameters& dtp) {
    const std::string space(" ");
    Tree::ByLine bl;
    is >> bl;
    auto vec = split(toupper(bl), space);
    for ( auto& v : vec ) {
      auto jev = split(v, "=");
      if ( dtp._enums.find(jev[0]) != dtp._enums.end() ) {
        if ( jev[0] == toupper("Criterion") ) {
          if ( jev[1] == "GINI" )
            dtp._criterion = Criterion::GINI;
          else if ( jev[1] == "ENTROPY" )
            dtp._criterion = Criterion::ENTROPY;
          else
            throw std::domain_error("Bad Criterion name: " + jev[1]);
        } else if ( jev[0] == toupper("SplitStrategy") ) {
          if ( jev[1] == "BEST" )
            dtp._splitStrategy = SplitStrategy::BEST;
          else if ( jev[1] == "RANDOM" )
            dtp._splitStrategy = SplitStrategy::RANDOM;
          else
            throw std::domain_error("Bad SplitStrategy name: " + jev[1]);
        } else if ( jev[0] == toupper("SPLITMAXFEAT") ) {
          if ( jev[1] == "INT" )
            dtp._splitMaxSelect = SplitMaxFeat::INT;
          else if ( jev[1] == "FLT" )
            dtp._splitMaxSelect = SplitMaxFeat::FLT;
          else if ( jev[1] == "SQRT" )
            dtp._splitMaxSelect = SplitMaxFeat::SQRT;
          else if ( jev[1] == "LOG2" )
            dtp._splitMaxSelect = SplitMaxFeat::LOG2;
          else
            throw std::domain_error("Bad SplitMaxFeat name: " + jev[1]);
        } else if ( jev[0] == toupper("ClassWeightType") ) {
          if ( jev[1] == "SAME" )
            dtp._classWeightType = ClassWeightType::SAME;
          else if ( jev[1] == "BALANCED" )
            dtp._classWeightType = ClassWeightType::BALANCED;
          else
            throw std::domain_error("Bad ClassWeightType name: " + jev[1]);
        } else {
          throw std::domain_error("Programming error: operator>> for DecisionTreeParameters");
        }
      } else {
        if ( jev[0] == toupper("MaxDepth") )
          dtp.set_value(dtp._maxDepth, jev[0], jev[1]);
        else if ( jev[0] == toupper("MaxLeafNodes") )
          dtp.set_value(dtp._maxLeafNodes, jev[0], jev[1]);
        else if ( jev[0] == toupper("MinLeafSamples") )
          dtp.set_value(dtp._minLeafSamples, jev[0], jev[1]);
        else if ( jev[0] == toupper("MinPuritySplit") )
          dtp.set_value(dtp._minPuritySplit, jev[0], jev[1]);
        else if ( jev[0] == toupper("UseZeroesInSplit") )
          dtp.set_value(dtp._useImpliedZerosInSplitDecision, jev[0], jev[1]);
        else if ( jev[0] == toupper("Seed") )
          dtp.set_value(dtp._seed, jev[0], jev[1]);
        else if ( jev[0] == toupper("SplitMaxFeatures") )
          dtp.set_value(dtp._splitMaxFeatures, jev[0], jev[1]);
        else if ( jev[0] == toupper("SelectSplitNumeric") )
          dtp.set_value(dtp._selectSplitNumeric, jev[0], jev[1]);
        else
          throw std::domain_error("model error: operator>> for DecisionTreeParameters");
      }
    } // for

    dtp._enums[toupper("Criterion")] = dtp.get_enum(dtp._criterion);
    dtp._enums[toupper("SplitStrategy")] = dtp.get_enum(dtp._splitStrategy);
    dtp._enums[toupper("SplitMaxFeat")] = dtp.get_enum(dtp._splitMaxSelect);
    dtp._enums[toupper("ClassWeightType")] = dtp.get_enum(dtp._classWeightType);

    dtp.do_seed();

    return is;
  }

private:
  void do_seed() {
    std::seed_seq ss{uint32_t(_seed & 0xffffffff), uint32_t(_seed>>32)};
    _rng.seed(ss);
  }

  template <typename T>
  int get_enum(T t) { return static_cast<int>(t); }

  template <typename T>
  void set_value(T& t, const std::string& s, const std::string& v) { t = check_value<T>(s,v); }

  enum struct Nums { BIGINT, PLUSBIGINT, PLUSBIGINT_NOZERO, INT, PLUSINT, PLUSINT_NOZERO, REAL, PLUSREAL, PLUSREAL_NOZERO, BOOL };

  /* basic stuff only:
      doesn't support scientific notation
      doesn't allow '+' out in front
      certainly others
' */
  template <typename T>
  T
  check_value(const std::string& s, const std::string& v) {
    static const std::string PlusInts = "0123456789";
    static const std::string Ints = "-" + PlusInts;
    static const std::string PlusReals = "." + PlusInts;
    static const std::string Reals = "." + Ints;
    static const std::string Bools = "01";
    auto f = _argmap.find(toupper(s));
    if ( f == _argmap.end() )
      throw std::domain_error("Unknown Option: " + s);

    switch (f->second) {
      case Nums::BIGINT:
        if ( v.find_first_not_of(Ints) != std::string::npos )
          throw std::domain_error("Unknown <long> numeric: " + v);
        else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
          throw std::domain_error("Bad signed long integer: " + v);
        return static_cast<T>(std::atol(v.c_str()));
      case Nums::PLUSBIGINT:
        if ( v.find_first_not_of(Ints) != std::string::npos )
          throw std::domain_error("Unknown <long> numeric: " + v);
        else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
          throw std::domain_error("Bad signed long integer: " + v);
        else if ( std::atof(v.c_str()) == .0 )
          throw std::domain_error("Bad big <+int> gt 0: " + v);
        return static_cast<T>(std::strtoul(v.c_str(), NULL, 10));
      case Nums::PLUSBIGINT_NOZERO:
        if ( v.find_first_not_of(Ints) != std::string::npos )
          throw std::domain_error("Unknown <long> numeric: " + v);
        else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
          throw std::domain_error("Bad signed long integer: " + v);
        return static_cast<T>(std::strtoul(v.c_str(), NULL, 10));
      case Nums::INT:
        if ( v.find_first_not_of(Ints) != std::string::npos )
          throw std::domain_error("Unknown <int> numeric: " + v);
        else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
          throw std::domain_error("Bad signed integer: " + v);
        return static_cast<T>(std::atoi(v.c_str()));
      case Nums::PLUSINT:
        if ( v.find_first_not_of(PlusInts) != std::string::npos )
          throw std::domain_error("Unknown <+int> numeric: " + v);
        return static_cast<T>(std::atoi(v.c_str()));
      case Nums::PLUSINT_NOZERO:
        if ( v.find_first_not_of(PlusInts) != std::string::npos )
          throw std::domain_error("Unknown <+int> numeric: " + v);
        else if ( std::atof(v.c_str()) == .0 )
          throw std::domain_error("Bad <+int> gt 0: " + v);
        return static_cast<T>(std::atoi(v.c_str()));
      case Nums::REAL:
        if ( v.find_first_not_of(Reals) != std::string::npos )
          throw std::domain_error("Unknown <floating> numeric: " + v);
        else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
          throw std::domain_error("Bad signed floating: " + v);
        else if ( std::count(v.begin(), v.end(), '.') > 1 )
          throw std::domain_error("Bad decimal: " + v);
        return static_cast<T>(std::atof(v.c_str()));
      case Nums::PLUSREAL:
        if ( v.find_first_not_of(PlusReals) != std::string::npos )
          throw std::domain_error("Unknown <floating> numeric: " + v);
        else if ( std::count(v.begin(), v.end(), '.') > 1 )
          throw std::domain_error("Bad decimal: " + v);
        return static_cast<T>(std::atof(v.c_str()));
      case Nums::PLUSREAL_NOZERO:
        if ( v.find_first_not_of(PlusReals) != std::string::npos )
          throw std::domain_error("Unknown <+floating> numeric: " + v);
        else if ( (T)0 == std::atof(v.c_str()) ) // set to 0 by atof if a bad numeric string
          throw std::domain_error("Bad <+floating> gt 0: " + v);
        else if ( std::count(v.begin(), v.end(), '.') > 1 )
          throw std::domain_error("Bad decimal: " + v);
        else if ( std::atof(v.c_str()) == .0 )
          throw std::domain_error("Bad <+int> gt 0: " + v);
        return static_cast<T>(std::atof(v.c_str()));
      case Nums::BOOL:
        if ( v.find_first_not_of(Bools) != std::string::npos )
          throw std::domain_error("Unknown <bool> numeric: " + v);
        return static_cast<T>(v[0] == '1');
      default:
        throw std::domain_error("program error: unknown numeric in dtp::set_value()");
    }
  }

private:
  friend struct DecisionTreeClassifier;

  // by construction (would be const if not for operator>> and my laziness to do it differently)
  Criterion _criterion;
  SplitStrategy _splitStrategy;
  SplitMaxFeat _splitMaxSelect; // if INT, then _selectSplitNumeric > 0.  if FLT, then 0 < _selectSplitNumeric <= 1
  int _maxDepth; // 0 means expand until all leaves are pure or until all nodes contain less than _minLeafSamples
  int _maxLeafNodes; // 0, then unlimited leaf nodes, else grow a tree until you reach _maxLeafNodes using bfs
  dtype _minLeafSamples; // if >= 1 - use int(), 0 - error, < 1 - use as percentage
  dtype _minPuritySplit; // A node will split if its purity is less than the threshold, otherwise it is a leaf
  bool _useImpliedZerosInSplitDecision;
  ClassWeightType _classWeightType;
  long int _seed;

  std::uniform_real_distribution<double> _unif;
  std::mt19937_64 _rng;

  // set after or while data are read
  dtype _splitMaxFeatures;
  dtype _selectSplitNumeric; // if _splitMaxSelect is INT,FLT then this is the value of the INT or FLT

  // not user-settable
  std::unordered_map<std::string, int> _enums; // support operator>>
  std::unordered_map<std::string, Nums> _argmap; // support operator>>
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
inline dtype
purity(const std::array<dtype, 2>& a) {
  return (std::max(a[0], a[1])/(a[0]+a[1]));
}

/*
  DecisionTreeClassifier
*/
struct DecisionTreeClassifier {
  explicit DecisionTreeClassifier(DecisionTreeParameters& dtp)
    : _dtp(dtp) {}

  std::list<label>
  classify(const DataMatrix& newData) {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before applying one!");
std::list<label> toRtn;
return toRtn;
  }

  label
  classify(const RowVector& row) const {
    if ( _nodes.empty() )
      throw std::domain_error("Need to learn a model before you can use one!");

    static constexpr std::size_t FID = 0;
    static constexpr std::size_t THOLD = 1;
    for ( const auto& i : _nodes ) {
      if ( std::get<THOLD>(*std::get<0>(i)) > row[std::get<FID>(*std::get<0>(i))]  ) // go left
std::cout << "not implemented" << std::endl;
    } // for
label huh;
return huh;
  }

private:
  static constexpr std::size_t QS = 0;
  static constexpr std::size_t TH = 1;
  static constexpr std::size_t LP = 2;
  static constexpr std::size_t RP = 3;

  typedef bool IsLeaf;
  typedef featureID FeatureID;
  typedef dtype SplitValue;
  typedef std::tuple<FeatureID, SplitValue> Core;
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
    //  'go to the right', left child purity and right child purity
    static constexpr std::size_t FID = 0;
    static constexpr std::size_t SPV = 1;
    static constexpr std::size_t MON = 2;
    static constexpr std::size_t LPU = 3;
    static constexpr std::size_t RPU = 4;
// sjn might make better sense to make the 3 monitors pointers everywhere; test performance later
    auto p = find_split(dm, nSplitFeatures, parentMonitor);
    Node current(new Core(std::get<FID>(p), std::get<SPV>(p)), nullptr, nullptr); // root
    decltype(p) q;

    // partition parent into left/right
    Monitor rightChildMonitor = std::get<MON>(p);
    Monitor leftChildMonitor = ~rightChildMonitor & parentMonitor;

    typedef std::tuple<Core*, Monitor, Monitor> internal;
    static constexpr std::size_t PARENT_CPTR = 0;
    static constexpr std::size_t LC_CPTR = 1, LC_MON = LC_CPTR;
    static constexpr std::size_t RC_CPTR = 2, RC_MON = RC_CPTR;
    std::queue<internal> que;
    que.push(std::make_tuple(std::get<PARENT_CPTR>(current), leftChildMonitor, rightChildMonitor));

    // breadth first traversal
    while ( !que.empty() ) {
      auto& e = que.front();
      current = Node(std::get<PARENT_CPTR>(e), nullptr, nullptr);
      p = find_split(dm, nSplitFeatures, std::get<LC_MON>(e));
      std::get<LC_CPTR>(current) = new Core(std::get<FID>(p), std::get<SPV>(p));

      q = find_split(dm, nSplitFeatures, std::get<RC_MON>(e));
      std::get<RC_CPTR>(current) = new Core(std::get<FID>(q), std::get<SPV>(q));
      _nodes.push_front(current);

      parentMonitor = std::get<LC_MON>(e);
      rightChildMonitor = std::get<MON>(p);
      leftChildMonitor = ~rightChildMonitor & parentMonitor;
      auto nextl = std::make_tuple(std::get<LC_MON>(current), leftChildMonitor, rightChildMonitor);

      parentMonitor = std::get<RC_MON>(e);
      rightChildMonitor = std::get<MON>(q);
      leftChildMonitor = ~rightChildMonitor & parentMonitor;
      auto nextr = std::make_tuple(std::get<RC_MON>(current), leftChildMonitor, rightChildMonitor);

// sjn shouldn't I use std::get<LPU>(p) here and std::get<RPU>(p) ?
      que.pop();
      if ( !done(dm, nextl) )
        que.push(nextl);
      if ( !done(dm, nextr) )
        que.push(nextr);
    } // while
  }

  bool
  done(const DataMatrix& dm, const std::tuple<Core*, Monitor, Monitor>& t) const {
return true;
  }

public:
  ~DecisionTreeClassifier() {
    auto jiter = _nodes.begin();
    if ( jiter != _nodes.end() ) {
      for ( auto iter = _nodes.begin(); iter != _nodes.end(); ) {
        if ( std::get<0>(*iter) )
          delete std::get<0>(*iter);
        if ( ++iter != _nodes.end() )
          ++jiter;
      } // for

      if ( std::get<1>(*jiter) )
        delete std::get<1>(*jiter);
      if ( std::get<2>(*jiter) )
        delete std::get<2>(*jiter);
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
    return input;
  }

  friend // write out this model
  std::ostream&
  operator<<(std::ostream& output, const DecisionTreeClassifier& dtc) {
    
    return output;
  }

protected:
  inline
  bool done() const {
return false;
  }

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
      if ( 0 == class_total ) // sjn : is this possible?
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
  std::tuple<dtype, dtype, dtype, dtype>
  split_gini(const DataMatrix& dm, std::size_t featColumn, SplitStrategy s, bool includeZeroes) {
    const auto& labels = ub::column(dm, 0);
    const auto& values = ub::column(dm, featColumn);
    auto iter = values.begin(), iter_end = values.end();
    std::tuple<dtype, dtype, dtype, dtype> rtn(0,0,0,0);
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
            if ( u > std::get<TH>(rtn) )
              nr[labels(i)]++;
            else
              nl[labels(i)]++;
          } // for
        } else { // only look at labels of rows with nonZero values for this featColumn
          for ( auto viter = values.begin(); viter != values.end(); ++viter ) { // these are const_iterators due to values
            if ( *viter > std::get<TH>(rtn) )
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
  std::tuple<dtype, dtype, dtype, dtype> /* quality, split-value, purity-left, purity-right */
  evaluate(const DataMatrix& dm, std::size_t featColumn) {
    switch(_dtp._criterion) {
      case Criterion::GINI:
        return split_gini(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision);
      case Criterion::ENTROPY:
//        return split_entropy(dm, featColumn, _dtp._splitStrategy, _dtp._useImpliedZerosInSplitDecision);
return std::make_tuple(0,0,0,0);
    }
  }

  // find_split() returns feature ID, split value and monitor showing which 'mask' bits
  //  go to the right (if set on input, stays set if to the right, unset otherwise)
  //  finally, it return the left and right purities of child nodes
  std::tuple<featureID, dtype, Monitor, dtype, dtype>
  find_split(const DataMatrix& dm, std::size_t nSplitFeatures, Monitor mask) {
    const std::size_t numFeatures = dm.size2();
    std::size_t featureCount = 0;
    dtype currBest = -1;
    dtype currValue = 0;
    dtype currLPurity = 0;
    dtype currRPurity = 0;
    featureID currCol = 0;
    std::set<std::size_t> selectedFeats;
    while ( featureCount < nSplitFeatures ) {
      featureID col = 0;
      while (true) {
// sjn: could loop forever on edge cases - fix this
        col = static_cast<featureID>(std::round(_dtp._unif(_dtp._rng) * numFeatures));
        if ( selectedFeats.insert(col).second && !allZeroes(dm, col, mask) )
          break;
      } // while
      auto val = evaluate(dm, col); // return split quality, split value, lpurity, rpurity
      if ( std::get<QS>(val) > currBest ) {
        currBest = std::get<QS>(val);
        currValue = std::get<TH>(val);
        currLPurity = std::get<LP>(val);
        currRPurity = std::get<RP>(val);
        currCol = col;
      }
      ++featureCount;
    } // while

    std::size_t beg = mask.find_first();
    if ( currCol == 0 || beg == mask.npos )
      throw std::domain_error("something very wrong: find_split()");

    const auto& vals = column(dm, currCol);
    const dtype threshold = currValue;
    while ( beg != mask.npos ) {
      if ( threshold > vals[beg] ) // goes to left child (could be implicit zero)
        mask[beg] = false;
      beg = mask.find_next(beg);
    } // while
    return std::make_tuple(currCol, threshold, mask, currLPurity, currRPurity);
  }

  DecisionTreeParameters& _dtp;
  std::forward_list<Node> _nodes;
};

} // namespace Tree

#endif // __ALTIUS_DTREE_HPP__
