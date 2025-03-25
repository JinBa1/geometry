#ifndef COMPUTE_MEAN_CURVATURE_FLOW_HEADER_FILE
#define COMPUTE_MEAN_CURVATURE_FLOW_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "compute_areas_normals.h"

void mean_curvature_flow(const Eigen::MatrixXi& F,
                         const Eigen::SparseMatrix<double>& L,
                         const double timeStep,
                         const Eigen::SparseMatrix<double>& M,
                         const Eigen::SparseMatrix<double>& MInv,
                         const Eigen::VectorXi& boundVMask,
                         const bool isExplicit,
                         Eigen::MatrixXd& currV){

    using namespace Eigen;
    using namespace std;

    // Save original vertex positions
    MatrixXd origV = currV;

    if (isExplicit) {
        // Explicit Euler integration: v(t + Δt) = v(t) - δt * M⁻¹ * L * v(t)
        MatrixXd Lv = L * currV;

        // Update non-boundary vertices
        for (int i = 0; i < currV.rows(); ++i) {
            if (boundVMask(i) == 0) {
                currV.row(i) -= timeStep * (MInv.coeff(i,i) * Lv.row(i));
            }
        }
    } else {
        // Implicit Euler integration: (M + δt * L) * v(t + Δt) = M * v(t)

        // Get the dimensions
        int numVertices = currV.rows();
        int numBoundary = boundVMask.sum();
        int numInterior = numVertices - numBoundary;

        if (numBoundary == 0) {
            // No boundary vertices - solve the full system directly
            SparseMatrix<double> A = M + timeStep * L;

            // Solve for each coordinate
            SimplicialLDLT<SparseMatrix<double>> solver;
            solver.compute(A);

            if (solver.info() != Success) {
                cerr << "Decomposition failed!" << endl;
                return;
            }

            MatrixXd rhs = M * currV;

            for (int j = 0; j < 3; ++j) {
                currV.col(j) = solver.solve(rhs.col(j));
            }
        } else {
            // Identify boundary and interior vertices
            vector<int> interiorIndices;
            for (int i = 0; i < numVertices; ++i) {
                if (boundVMask(i) == 0) {
                    interiorIndices.push_back(i);
                }
            }

            // Extract submatrices for interior vertices only
            SparseMatrix<double> L_ii(numInterior, numInterior);
            SparseMatrix<double> M_ii(numInterior, numInterior);
            MatrixXd V_i(numInterior, 3);

            // Create index mapping
            vector<int> intToFullMap(numInterior);
            map<int, int> fullToIntMap;

            for (int i = 0; i < numInterior; ++i) {
                intToFullMap[i] = interiorIndices[i];
                fullToIntMap[interiorIndices[i]] = i;
            }

            // Fill interior vertex positions
            for (int i = 0; i < numInterior; ++i) {
                V_i.row(i) = currV.row(intToFullMap[i]);
            }

            // Extract L_ii and M_ii (interior-interior blocks)
            vector<Triplet<double>> L_ii_triplets;
            vector<Triplet<double>> M_ii_triplets;

            // Process L matrix
            for (int k = 0; k < L.outerSize(); ++k) {
                for (SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();

                    // Only include if both row and column are interior vertices
                    if (boundVMask(row) == 0 && boundVMask(col) == 0) {
                        L_ii_triplets.push_back(Triplet<double>(
                            fullToIntMap[row],
                            fullToIntMap[col],
                            it.value()
                        ));
                    }
                }
            }

            // Process M matrix (diagonal)
            for (int i = 0; i < numInterior; ++i) {
                int fullIdx = intToFullMap[i];
                M_ii_triplets.push_back(Triplet<double>(
                    i, i, M.coeff(fullIdx, fullIdx)
                ));
            }

            // Set matrices from triplets
            L_ii.setFromTriplets(L_ii_triplets.begin(), L_ii_triplets.end());
            M_ii.setFromTriplets(M_ii_triplets.begin(), M_ii_triplets.end());

            // Construct systems matrix for interior vertices: A_ii = M_ii + timeStep * L_ii
            SparseMatrix<double> A_ii = M_ii + timeStep * L_ii;

            // Compute right-hand side for interior vertices: M_ii * V_i
            MatrixXd rhs_i = M_ii * V_i;

            // Now extract L_ib and compute effect of boundary vertices on interior vertices
            MatrixXd L_ib_V_b = MatrixXd::Zero(numInterior, 3);

            // Loop through L to find connections between interior and boundary vertices
            for (int k = 0; k < L.outerSize(); ++k) {
                for (SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();

                    // Interior-Boundary interaction
                    if (boundVMask(row) == 0 && boundVMask(col) == 1) {
                        L_ib_V_b.row(fullToIntMap[row]) += it.value() * currV.row(col);
                    }
                }
            }

            // Adjust right-hand side: rhs_i = rhs_i - timeStep * L_ib * V_b
            rhs_i -= timeStep * L_ib_V_b;

            // Solve the system for interior vertices
            SimplicialLDLT<SparseMatrix<double>> solver;
            solver.compute(A_ii);

            if (solver.info() != Success) {
                cerr << "Decomposition failed!" << endl;
                return;
            }

            // Solve for each coordinate
            MatrixXd V_i_new(numInterior, 3);
            for (int j = 0; j < 3; ++j) {
                V_i_new.col(j) = solver.solve(rhs_i.col(j));

                if (solver.info() != Success) {
                    cerr << "Solving failed for dimension " << j << "!" << endl;
                    return;
                }
            }

            // Copy results back to full system
            for (int i = 0; i < numInterior; ++i) {
                currV.row(intToFullMap[i]) = V_i_new.row(i);
            }
        }
    }

    // If there are no boundary vertices, center and rescale the mesh
    if (boundVMask.sum() == 0) {
        // Calculate original mesh area
        VectorXd faceAreas;
        MatrixXd faceNormals;
        compute_areas_normals(origV, F, faceAreas, faceNormals);
        double originalArea = faceAreas.sum();

        // Center the mesh
        Vector3d centroid = currV.colwise().mean();
        currV.rowwise() -= centroid.transpose();

        // Calculate new mesh area
        compute_areas_normals(currV, F, faceAreas, faceNormals);
        double newArea = faceAreas.sum();

        // Scale to preserve area
        if (newArea > 1e-10) {
            double scaleFactor = sqrt(originalArea / newArea);
            currV *= scaleFactor;
        }
    }
}

#endif