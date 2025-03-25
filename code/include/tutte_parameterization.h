#ifndef COMPUTE_TUTTE_PARAMETERIZATION_HEADER_FILE
#define COMPUTE_TUTTE_PARAMETERIZATION_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "slice_columns_sparse.h"
#include "set_diff.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Eigen::MatrixXd compute_boundary_embedding(const Eigen::MatrixXd& V,
                                           const Eigen::VectorXi& boundVertices,
                                           const double r){

    using namespace Eigen;

    // Create the result matrix for UV coordinates of boundary vertices
    MatrixXd UVBound = MatrixXd::Zero(boundVertices.size(), 2);

    // Calculate edge lengths along the boundary
    double totalLength = 0.0;
    std::vector<double> edgeLengths(boundVertices.size());

    for (int i = 0; i < boundVertices.size(); ++i) {
        int currentIdx = boundVertices(i);
        int nextIdx = boundVertices((i + 1) % boundVertices.size());

        // Calculate edge length
        double length = (V.row(nextIdx) - V.row(currentIdx)).norm();
        edgeLengths[i] = length;
        totalLength += length;
    }

    // Calculate sector angles (psi) based on relative edge lengths
    std::vector<double> sectorAngles(boundVertices.size());
    for (int i = 0; i < boundVertices.size(); ++i) {
        sectorAngles[i] = 2.0 * M_PI * (edgeLengths[i] / totalLength);
    }

    // First boundary vertex is placed at (r, 0)
    UVBound(0, 0) = r;
    UVBound(0, 1) = 0.0;

    // Place remaining vertices on the circle using cumulative angles
    double cumulativeAngle = 0.0;
    for (int i = 1; i < boundVertices.size(); ++i) {
        cumulativeAngle += sectorAngles[i-1];

        // Place vertex using polar coordinates
        UVBound(i, 0) = r * cos(cumulativeAngle);
        UVBound(i, 1) = r * sin(cumulativeAngle);
    }

    return UVBound;
}

Eigen::MatrixXd compute_tutte_embedding(const Eigen::VectorXi& boundVertices,
                                        const Eigen::MatrixXd& UVBound,
                                        const Eigen::SparseMatrix<double>& d0,
                                        const Eigen::SparseMatrix<double>& W){

    using namespace Eigen;

    // Total number of vertices is the number of columns in d0
    int totalVertices = d0.cols();

    // Identify interior vertex indices
    VectorXi allVertices = VectorXi::LinSpaced(totalVertices, 0, totalVertices - 1);
    VectorXi interiorVertices = set_diff(allVertices, boundVertices);

    // Slice d0 into boundary and interior parts
    SparseMatrix<double> d0B = slice_columns_sparse(d0, boundVertices);
    SparseMatrix<double> d0I = slice_columns_sparse(d0, interiorVertices);

    // Construct the linear system for the Laplacian equation
    // d0I^T * W * d0I * UVI = -d0I^T * W * d0B * UVB
    SparseMatrix<double> A = d0I.transpose() * W * d0I;
    MatrixXd B = -d0I.transpose() * W * d0B * UVBound;

    // Solve the system for interior UV coordinates
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    // Check if decomposition succeeded
    if (solver.info() != Eigen::Success) {
        // In case of failure, return empty result
        std::cerr << "Decomposition failed!" << std::endl;
        return MatrixXd::Zero(totalVertices, 2);
    }

    // Solve for the interior UV coordinates
    MatrixXd UVI = solver.solve(B);

    // Create the full UV result matrix
    MatrixXd UV = MatrixXd::Zero(totalVertices, 2);

    // Fill in the boundary vertices
    for (int i = 0; i < boundVertices.size(); ++i) {
        UV.row(boundVertices(i)) = UVBound.row(i);
    }

    // Fill in the interior vertices
    for (int i = 0; i < interiorVertices.size(); ++i) {
        UV.row(interiorVertices(i)) = UVI.row(i);
    }

    return UV;
}

#endif