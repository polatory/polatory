// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <Eigen/Core>

typedef std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> points_normals;
typedef std::pair<std::vector<Eigen::Vector3d>, Eigen::VectorXd> points_values;

inline
points_normals read_points_and_normals(const std::string& filename)
{
   std::ifstream ifs(filename);
   if (!ifs)
      return points_normals();

   std::vector<Eigen::Vector3d> points;
   std::vector<Eigen::Vector3d> normals;

   std::string line;

   while (std::getline(ifs, line)) {
      if (boost::starts_with(line, "#"))
         continue;

      std::vector<std::string> row;
      boost::split(row, line, boost::is_any_of(" \t,"));
      if (row.size() < 6)
         continue;

      auto x = std::stod(row[0]);
      auto y = std::stod(row[1]);
      auto z = std::stod(row[2]);
      auto nx = std::stod(row[3]);
      auto ny = std::stod(row[4]);
      auto nz = std::stod(row[5]);

      points.push_back(Eigen::Vector3d(x, y, z));
      normals.push_back(Eigen::Vector3d(nx, ny, nz));
   }

   return std::make_pair(std::move(points), std::move(normals));
}


inline
points_values read_points_and_values(const std::string& filename)
{
   std::ifstream ifs(filename);
   if (!ifs)
      return points_values();

   std::vector<Eigen::Vector3d> points;
   std::vector<double> values_tmp;

   std::string line;

   while (std::getline(ifs, line)) {
      if (boost::starts_with(line, "#"))
         continue;

      std::vector<std::string> row;
      boost::split(row, line, boost::is_any_of(" \t,"));
      if (row.size() < 4)
         continue;

      auto x = std::stod(row[0]);
      auto y = std::stod(row[1]);
      auto z = std::stod(row[2]);
      auto value = std::stod(row[3]);

      points.push_back(Eigen::Vector3d(x, y, z));
      values_tmp.push_back(value);
   }

   Eigen::VectorXd values = Eigen::VectorXd::Map(values_tmp.data(), values_tmp.size());

   return std::make_pair(std::move(points), std::move(values));
}
