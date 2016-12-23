/*
  FILE: utils.hpp
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

#ifndef __ALTIUS_SIMPLE_UTILS_HPP__
#define __ALTIUS_SIMPLE_UTILS_HPP__

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace Utils {

enum struct Nums { BIGINT, PLUSBIGINT, PLUSBIGINT_NOZERO, INT, PLUSINT, PLUSINT_NOZERO, REAL, PLUSREAL, PLUSREAL_NOZERO, BOOL };

/* basic stuff only:
    doesn't support scientific notation
    doesn't allow '+' out in front
    certainly others
*/
template <typename T>
T convert(const std::string& v, Nums constraint) {
  static const std::string PlusInts = "0123456789";
  static const std::string Ints = "-" + PlusInts;
  static const std::string PlusReals = "." + PlusInts;
  static const std::string Reals = "." + Ints;
  static const std::string Bools = "01";

  switch (constraint) {
    case Nums::BIGINT:
      if ( v.find_first_not_of(Ints) != std::string::npos )
        throw std::domain_error("Unknown <long> numeric: " + v);
      else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
        throw std::domain_error("Bad signed long integer: " + v);
      break;
    case Nums::PLUSBIGINT:
      if ( v.find_first_not_of(Ints) != std::string::npos )
        throw std::domain_error("Unknown <long> numeric: " + v);
      else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
        throw std::domain_error("Bad signed long integer: " + v);
      else if ( std::atof(v.c_str()) == .0 )
        throw std::domain_error("Bad big <+int> gt 0: " + v);
      break;
    case Nums::PLUSBIGINT_NOZERO:
      if ( v.find_first_not_of(Ints) != std::string::npos )
        throw std::domain_error("Unknown <long> numeric: " + v);
      else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
        throw std::domain_error("Bad signed long integer: " + v);
      break;
    case Nums::INT:
      if ( v.find_first_not_of(Ints) != std::string::npos )
        throw std::domain_error("Unknown <int> numeric: " + v);
      else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
        throw std::domain_error("Bad signed integer: " + v);
      break;
    case Nums::PLUSINT:
      if ( v.find_first_not_of(PlusInts) != std::string::npos )
        throw std::domain_error("Unknown <+int> numeric: " + v);
      break;
    case Nums::PLUSINT_NOZERO:
      if ( v.find_first_not_of(PlusInts) != std::string::npos )
        throw std::domain_error("Unknown <+int> numeric: " + v);
      else if ( std::atof(v.c_str()) == .0 )
        throw std::domain_error("Bad <+int> gt 0: " + v);
      break;
    case Nums::REAL:
      if ( v.find_first_not_of(Reals) != std::string::npos )
        throw std::domain_error("Unknown <floating> numeric: " + v);
      else if ( v.find("-") != std::string::npos && v.find("-") != 0 )
        throw std::domain_error("Bad signed floating: " + v);
      else if ( std::count(v.begin(), v.end(), '.') > 1 )
        throw std::domain_error("Bad decimal: " + v);
      break;
    case Nums::PLUSREAL:
      if ( v.find_first_not_of(PlusReals) != std::string::npos )
        throw std::domain_error("Unknown <floating> numeric: " + v);
      else if ( std::count(v.begin(), v.end(), '.') > 1 )
        throw std::domain_error("Bad decimal: " + v);
      break;
    case Nums::PLUSREAL_NOZERO:
      if ( v.find_first_not_of(PlusReals) != std::string::npos )
        throw std::domain_error("Unknown <+floating> numeric: " + v);
      else if ( (T)0 == std::atof(v.c_str()) ) // set to 0 by atof if a bad numeric string
        throw std::domain_error("Bad <+floating> gt 0: " + v);
      else if ( std::count(v.begin(), v.end(), '.') > 1 )
        throw std::domain_error("Bad decimal: " + v);
      else if ( std::atof(v.c_str()) == .0 )
        throw std::domain_error("Bad <+int> gt 0: " + v);
      break;
    case Nums::BOOL:
      if ( v.find_first_not_of(Bools) != std::string::npos )
        throw std::domain_error("Unknown <bool> numeric: " + v);
      break;
    default:
      throw std::domain_error("program error: unknown numeric in dtp::set_value()");
  }

  std::stringstream c(v);
  T t;
  c >> t;
  return t;
}

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

std::string uppercase(const std::string& s) {
  std::string rtn = s;
  for ( std::size_t i = 0; i < rtn.size(); ++i )
    rtn[i] = std::toupper(rtn[i]);
  return rtn;
}

} // namespace Utils

#endif // __ALTIUS_SIMPLE_UTILS_HPP__
