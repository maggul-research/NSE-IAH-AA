/* ---------------------------------------------------------------------
 * Copyright (C) 2025
 * Authored by Mustafa Aggul and Sinan Ergen
 * Hacettepe University && Southern Methodist University
 * Hacettepe University && Balikesir University
 *
 * This is the implementation file for the accuracy analysis of
 * the improved Arrow Hurwicz Method with Anderson Acceleration
 * for Navier Stokes Equation.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace MyNSE {
using namespace dealii;
using namespace std;

template <int dim> class AccuracyTest {
public:
  AccuracyTest(const unsigned int degree, const double Re, const double rho,
               const double alpha, const int m);
  void run(const unsigned int n_refinements);
  unsigned int get_final_iter() const { return final_iter; }

  /* Getters exposing the errors of the final (converged) state so that
     they can be written to the CSV file per (N, m) combination. */
  double get_velocity_error_l2() const { return final_velocity_error_l2; }
  double get_velocity_error_h1() const { return final_velocity_error_h1; }
  double get_pressure_error_l2() const { return final_pressure_error_l2; }
  double get_divergence_error_l2() const { return final_divergence_error_l2; }

private:
  void mesh_jobs(const unsigned int n_refinements);
  void setup_dofs();
  void initialize_system();
  /* assemble_id determines what parts of the system will be updated
     assemble_id: 0 assembles the system_matrix and system_rhs
     assemble_id: 1 updates the system_rhs with the current velocity
  */
  void assemble(int assemble_id);
  /* solve_id determines what parts of the system will be solved
     solve_id: 0 solves for the velocity
     solve_id: 1 solves for the pressure
  */
  void solve(int solve_id);
  void output_results(const unsigned int output_index) const;
  void compute_errors(BlockVector<double> AA_solution);
  void compute_divergence_error();
  void run_iterations(const double tolerance, const unsigned int max_iteration,
                      const bool output_result);

  void Anderson_Acceleration(FullMatrix<double> &F_temp,
                             FullMatrix<double> &u_tilde_temp,
                             BlockVector<double> &solution,
                             BlockVector<double> &old_solution,
                             BlockVector<double> &AA_solution,
                             double &norm_value, int &AA_iter, const int &m);

  const unsigned int degree;
  const double Re;
  const double rho;
  const double alpha;
  const int m;
  unsigned int final_iter;

  /* Errors of the final (converged) state, stored for CSV output. */
  double final_velocity_error_l2 = 0.0;
  double final_velocity_error_h1 = 0.0;
  double final_pressure_error_l2 = 0.0;
  double final_divergence_error_l2 = 0.0;

  const unsigned int solver_type;

  vector<types::global_dof_index> dofs_per_block;

  Triangulation<dim> triangulation;
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  BlockSparsityPattern sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> old_solution;
  BlockVector<double> AA_old_solution;
  BlockVector<double> AA_solution;
  BlockVector<double> solution;
  BlockVector<double> system_rhs;
  BlockVector<double> residue_vector;
};

/* Exact solution for the accuracy analysis */
template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>(dim + 1) {}
  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &value) const override;

private:
  const double theta = 1.0;
};

template <int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &values) const {
  const double x = p[0];
  const double y = p[1];

  values[0] =
      2 * theta * pow((pow(x, 2) - x), 2) * (pow(y, 2) - y) * (2 * y - 1);
  values[1] =
      -2 * theta * pow((pow(y, 2) - y), 2) * (pow(x, 2) - x) * (2 * x - 1);
  values[2] = theta * (2 * x - 1) * (2 * y - 1);
}

/* RHS based on the PDE and the given exact solution */
template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &value) const override;

private:
  const double theta = 1.0;
  const double Re = 1.0;
};

template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &values) const {
  const double x = p[0];
  const double y = p[1];
  values(0) = 4 * pow(theta, 2) * pow(pow(x, 2) - x, 3) * (2 * x - 1) *
                  pow(pow(y, 2) - y, 2) * (2 * pow(y, 2) - 2 * y + 1) -
              Re * (4 * theta * (pow(y, 2) - y) * (2 * y - 1) *
                        (6 * pow(x, 2) - 6 * x + 1) +
                    12 * theta * pow(pow(x, 2) - x, 2) * (2 * y - 1)) +
              2 * theta * (2 * y - 1);

  values(1) = 4 * pow(theta, 2) * pow(pow(y, 2) - y, 3) * (2 * y - 1) *
                  pow(pow(x, 2) - x, 2) * (2 * pow(x, 2) - 2 * x + 1) +
              Re * (4 * theta * (pow(x, 2) - x) * (2 * x - 1) *
                        (6 * pow(y, 2) - 6 * y + 1) +
                    12 * theta * pow(pow(y, 2) - y, 2) * (2 * x - 1)) +
              2 * theta * (2 * x - 1);
}

/* Exact velocity gradient for the accuracy analysis */
template <int dim> class ExactVelocityGradient : public Function<dim> {
public:
  ExactVelocityGradient() : Function<dim>(dim + 1) {}
  virtual void
  vector_gradient(const Point<dim> &p,
                  std::vector<Tensor<1, dim>> &gradient_values) const override;

private:
  const double theta = 1.0;
};

template <int dim>
void ExactVelocityGradient<dim>::vector_gradient(
    const Point<dim> &p, std::vector<Tensor<1, dim>> &gradient_values) const {
  const double x = p[0];
  const double y = p[1];

  for (unsigned int i = 0; i < dim; i++)
    gradient_values[i].clear();

  gradient_values[0][0] =
      4 * theta * (pow(x, 2) - x) * (2 * x - 1) * (pow(y, 2) - y) * (2 * y - 1);
  gradient_values[0][1] =
      2 * theta * pow(pow(x, 2) - x, 2) * (6 * pow(y, 2) - 6 * y + 1);
  gradient_values[1][0] =
      -2 * theta * pow(pow(y, 2) - y, 2) * (6 * pow(x, 2) - 6 * x + 1);
  gradient_values[1][1] = -4 * theta * (pow(y, 2) - y) * (2 * y - 1) *
                          (pow(x, 2) - x) * (2 * x - 1);
}

template <int dim>
AccuracyTest<dim>::AccuracyTest(const unsigned int degree, const double Re,
                                const double rho, const double alpha,
                                const int m)
    : degree(degree), Re(Re), rho(rho), alpha(alpha), m(m),
      solver_type(1) /* GMRES:0 and UMFPACK:1 */
      ,
      triangulation(Triangulation<dim>::maximum_smoothing),
      fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)

      ,
      dof_handler(triangulation) {}

template <int dim>
void AccuracyTest<dim>::mesh_jobs(const unsigned int n_refinements) {
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(n_refinements);
}

template <int dim> void AccuracyTest<dim>::setup_dofs() {
  system_matrix.clear();

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  unsigned int dof_u = dofs_per_block[0];
  unsigned int dof_p = dofs_per_block[1];

  FEValuesExtractors::Vector velocities(0);
  {
    constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler, 0,
                                             ExactSolution<dim>(), constraints,
                                             fe.component_mask(velocities));

    VectorTools::interpolate_boundary_values(dof_handler, 0,
                                             ExactSolution<dim>(), constraints,
                                             fe.component_mask(velocities));
  }
  constraints.close();

  cout << "   Number of active cells: " << triangulation.n_active_cells()
       << endl
       << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
       << dof_u << '+' << dof_p << ')' << endl;
}

template <int dim> void AccuracyTest<dim>::initialize_system() {
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit(sparsity_pattern);
  AA_old_solution.reinit(dofs_per_block);
  old_solution.reinit(dofs_per_block);
  solution.reinit(dofs_per_block);
  AA_solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
  residue_vector.reinit(dofs_per_block);

  solution = 0;
  AA_solution = 0;
}

template <int dim> void AccuracyTest<dim>::assemble(int assemble_id) {
  if (assemble_id == 0) {
    system_matrix = 0;
    system_rhs = 0;
  }

  QGauss<dim> quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs(dofs_per_cell);

  vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  RightHandSide<dim> right_hand_side;
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));

  vector<Tensor<1, dim>> velocity_values(n_q_points);
  vector<Tensor<2, dim>> velocity_gradients(n_q_points);
  vector<double> velocity_div(n_q_points);
  vector<double> pressure_values(n_q_points);
  vector<double> div_phi_u(dofs_per_cell);
  vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  vector<double> phi_p(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);

    local_matrix = 0;
    local_rhs = 0;

    if (assemble_id == 0) {
      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                        rhs_values);

      fe_values[velocities].get_function_values(solution, velocity_values);

      fe_values[velocities].get_function_gradients(solution,
                                                   velocity_gradients);

      fe_values[velocities].get_function_divergences(solution, velocity_div);

      fe_values[pressure].get_function_values(solution, pressure_values);
    }

    if (assemble_id == 1) {
      fe_values[velocities].get_function_divergences(solution, velocity_div);
    }

    for (unsigned int q = 0; q < n_q_points; ++q) {

      Tensor<1, dim> rhs_force;
      if (assemble_id == 0) {
        for (unsigned int d = 0; d < dim; ++d)
          rhs_force[d] = rhs_values[q](d);
      }

      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
        if (assemble_id == 0) {
          div_phi_u[k] = fe_values[velocities].divergence(k, q);
          grad_phi_u[k] = fe_values[velocities].gradient(k, q);
          phi_u[k] = fe_values[velocities].value(k, q);
          phi_p[k] = fe_values[pressure].value(k, q);
        }

        if (assemble_id == 1)
          phi_p[k] = fe_values[pressure].value(k, q);
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        if (assemble_id == 0)
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            local_matrix(i, j) +=
                ((1.0 / rho + 1.0 / Re) *
                     scalar_product(grad_phi_u[j], grad_phi_u[i])

                 + 1.0 * grad_phi_u[j] * velocity_values[q] * phi_u[i] +
                 0.5 * velocity_div[q] * phi_u[j] * phi_u[i]

                 + (rho / alpha) * div_phi_u[i] * div_phi_u[j]

                 + (alpha)*phi_p[j] * phi_p[i]

                 ) *
                fe_values.JxW(q);
          } // end dof iterations for phi_j

        if (assemble_id == 0)
          local_rhs(i) +=
              (1.0 / rho * scalar_product(velocity_gradients[q], grad_phi_u[i])

               + pressure_values[q] * div_phi_u[i]

               + (alpha)*pressure_values[q] * phi_p[i] + rhs_force * phi_u[i]

               ) *
              fe_values.JxW(q);

        else if (assemble_id == 1)
          local_rhs(i) -= (rho)*velocity_div[q] * phi_p[i] * fe_values.JxW(q);
      } // end dof iterations for phi_i
    } // end quadrature points iteration

    cell->get_dof_indices(local_dof_indices);

    if (assemble_id == 0)
      constraints.distribute_local_to_global(local_matrix, local_rhs,
                                             local_dof_indices, system_matrix,
                                             system_rhs);
    else
      constraints.distribute_local_to_global(local_rhs, local_dof_indices,
                                             system_rhs);
  } // end cell iteration
} // end assemble

template <int dim> void AccuracyTest<dim>::solve(int solve_id) {
  int blk = 3; // dummy choice 3
  switch (solve_id) {
  case 0:
    blk = 0; // solve for velocity corresponds to block 0
    break;
  case 1:
    blk = 1; // solve for pressure corresponds to block 1
    break;
  default:
    cout << " Unknown solve ID " << endl;
    break;
  }

  if (solver_type == 0) {
    SolverControl solver_control(100000, 1e-6, true);
    SolverGMRES<Vector<double>> gmres(solver_control);

    gmres.solve(system_matrix.block(blk, blk), solution.block(blk),
                system_rhs.block(blk), PreconditionIdentity());
    cout << " **GMRES steps: " << solver_control.last_step() << endl;
  } else if (solver_type == 1) {
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix.block(blk, blk));
    A_direct.vmult(solution.block(blk), system_rhs.block(blk));
  } else {
    cout << " Unknown solver type " << endl;
  }

  constraints.distribute(solution);
}

template <int dim>
void AccuracyTest<dim>::Anderson_Acceleration(FullMatrix<double> &F_temp,
                                              FullMatrix<double> &u_tilde_temp,
                                              BlockVector<double> &solution,
                                              BlockVector<double> &old_solution,
                                              BlockVector<double> &AA_solution,
                                              double &norm_value, int &AA_iter,
                                              const int &m) {
  FullMatrix<double> F(dof_handler.n_dofs(), AA_iter);
  FullMatrix<double> u_tilde(dof_handler.n_dofs(), AA_iter);
  for (int j = 0; j < m; j++) {
    F_temp.swap_col(m - j, m - 1 - j);
    u_tilde_temp.swap_col(m - j, m - 1 - j);
  }

  for (long unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
    F_temp(i, 0) = solution(i) - old_solution(i);
    u_tilde_temp(i, 0) = solution(i);

    for (int j = 0; j < AA_iter; j++) {
      F(i, j) = F_temp(i, j);
      u_tilde(i, j) = u_tilde_temp(i, j);
    }
  }

  if (AA_iter > 1) {
    Vector<double> alpha(AA_iter);

    if (AA_iter == 2) {
      double numerator = 0;
      double denominator = 0;

      for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        numerator += (F(i, 0) - F(i, 1)) * F(i, 1);
        denominator += (F(i, 0) - F(i, 1)) * (F(i, 0) - F(i, 1));
      }

      alpha(0) = -numerator / denominator;
      alpha(1) = 1 - alpha(0);
    } else {
      int m = AA_iter - 1;

      FullMatrix<double> sym_mat(m, m);
      Vector<double> rhs(m);
      Vector<double> temp(dof_handler.n_dofs());
      Vector<double> FtM(dof_handler.n_dofs());
      for (int i = 0; i < m; i++) {
        for (unsigned int k = 0; k < dof_handler.n_dofs(); k++) {
          rhs(i) -= (F(k, i) - F(k, m)) * F(k, m);

          for (int j = 0; j < m; j++) {
            sym_mat(i, j) += (F(k, i) - F(k, m)) * (F(k, j) - F(k, m));
          }
        }
      }

      FullMatrix<double> sym_mat_inv(m, m);
      sym_mat_inv.invert(sym_mat);

      Vector<double> alpha_new(m);
      sym_mat_inv.vmult(alpha_new, rhs);

      double sum = 0;
      for (int i = 0; i < m; i++) {
        alpha(i) = alpha_new(i);
        sum += alpha(i);
      }
      alpha(m) = 1 - sum;
    }

    Vector<double> AA_sol(dof_handler.n_dofs());
    u_tilde.vmult(AA_sol, alpha);
    AA_solution = AA_sol;
    norm_value = 0.0;
  }
}

template <int dim>
void AccuracyTest<dim>::run_iterations(const double tolerance,
                                       const unsigned int max_iteration,
                                       const bool output_result) {
  int AA_iter = 1;
  unsigned int th_iteration = 0;
  bool first_step = true;

  double current_res = 1000.0;

  // const int m = 2;
  double norm_value1;

  FullMatrix<double> F_temp(dof_handler.n_dofs(), m + 1);
  FullMatrix<double> u_tilde_temp(dof_handler.n_dofs(), m + 1);

  while ((first_step || (current_res > tolerance)) &&
         th_iteration < max_iteration) {

    old_solution = AA_solution;

    if ((th_iteration >= 1) && (th_iteration <= 1)) {
      old_solution = solution;
    } else if (th_iteration > 1) {
      solution = AA_solution;
    }

    assemble(0);
    solve(0);

    assemble(1);
    solve(1);

    first_step = false;

    if (th_iteration >= 0 && m != 0) {
      Anderson_Acceleration(F_temp, u_tilde_temp, solution, old_solution,
                            AA_solution, norm_value1, AA_iter, m);

      if (AA_iter < m + 1)
        AA_iter++;
    }

    if (AA_solution.l2_norm() < 1e-8) {
      residue_vector = solution;
      residue_vector -= old_solution;

      current_res =
          max(residue_vector.block(0).l2_norm() / solution.block(0).l2_norm(),
              residue_vector.block(1).l2_norm() / solution.block(1).l2_norm());

      cout << "**********" << endl;
      cout << " The relative error of the current iteration = " << current_res
           << " at " << " at " << th_iteration << "th iteration" << endl;

      std::cout << "**"
                << residue_vector.block(0).l2_norm() /
                       solution.block(0).l2_norm()
                << "**"
                << residue_vector.block(1).l2_norm() /
                       solution.block(1).l2_norm()
                << std::endl;
      std::cout << std::endl;
    } else {
      residue_vector = AA_solution;
      residue_vector -= old_solution;

      current_res = max(
          residue_vector.block(0).l2_norm() / AA_solution.block(0).l2_norm(),
          residue_vector.block(1).l2_norm() / AA_solution.block(1).l2_norm());

      std::cout << "**********" << std::endl;
      std::cout << " The relative error of AA iteration = " << current_res
                << " at " << " at " << th_iteration << "th iteration"
                << std::endl;
      std::cout << "**********" << std::endl;

      std::cout << "**"
                << residue_vector.block(0).l2_norm() /
                       AA_solution.block(0).l2_norm()
                << "**"
                << residue_vector.block(1).l2_norm() /
                       AA_solution.block(1).l2_norm()
                << std::endl;
      std::cout << std::endl;
    }

    ++th_iteration;

    if (output_result)
      output_results(th_iteration);

    if (current_res > 100.0) {
      th_iteration = 1000;
      break;
    }
    final_iter = th_iteration;
  }

  compute_errors(AA_solution);
}

template <int dim>
void AccuracyTest<dim>::compute_errors(BlockVector<double> AA_solution) {
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                   dim + 1);

  const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);

  Vector<double> cellwise_errors(triangulation.n_active_cells());
  QGauss<dim> quadrature(2 * degree + 3);

  ExactSolution<dim> exact_solution;

  VectorTools::integrate_difference(dof_handler, AA_solution, exact_solution,
                                    cellwise_errors, quadrature,
                                    VectorTools::L2_norm, &pressure_mask);
  const double current_pressure_error = VectorTools::compute_global_error(
      triangulation, cellwise_errors, VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler, AA_solution, exact_solution,
                                    cellwise_errors, quadrature,
                                    VectorTools::L2_norm, &velocity_mask);
  const double current_velocity_error = VectorTools::compute_global_error(
      triangulation, cellwise_errors, VectorTools::L2_norm);

  ExactVelocityGradient<dim> exact_solution_gradient;

  VectorTools::integrate_difference(
      dof_handler, AA_solution, exact_solution_gradient, cellwise_errors,
      quadrature, VectorTools::H1_seminorm, &velocity_mask);
  const double current_velocity_error_h1 = VectorTools::compute_global_error(
      triangulation, cellwise_errors, VectorTools::L2_norm);

  std::cout << "Errors: ||e_u||_L2 = " << current_velocity_error
            << ",  ||e_u||_H1 = " << current_velocity_error_h1
            << ",  ||e_p||_L2 = " << current_pressure_error << std::endl;
  std::cout << std::endl;

  // compute divergence error (divergance L2 norm)
  FEValues<dim> fe_values_div(fe, quadrature,
                              update_gradients | update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);
  std::vector<double> div_uh(
      quadrature.size()); // hold the divergence values at quadrature points
  double global_div_error = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values_div.reinit(cell);
    fe_values_div[velocities].get_function_divergences(AA_solution, div_uh);

    for (unsigned int q = 0; q < quadrature.size(); ++q) {
      global_div_error += (div_uh[q] * div_uh[q]) * fe_values_div.JxW(q);
    }
  }
  const double l2_divergence_error = std::sqrt(global_div_error);

  final_velocity_error_l2 = current_velocity_error;
  final_velocity_error_h1 = current_velocity_error_h1;
  final_pressure_error_l2 = current_pressure_error;
  final_divergence_error_l2 = l2_divergence_error;

  std::cout << "          || div(u_h) ||_L2 = " << l2_divergence_error
            << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;
} // end compute_errors

template <int dim>
void AccuracyTest<dim>::output_results(const unsigned int output_index) const {
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(AA_solution, solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  ostringstream filename;
  filename << "Re_" << Re << "solution"
           << Utilities::int_to_string(output_index, 4) << ".vtk";

  ofstream output(filename.str().c_str());
  data_out.write_vtk(output);
}

template <int dim>
void AccuracyTest<dim>::run(const unsigned int n_refinements) {
  mesh_jobs(n_refinements);
  setup_dofs();
  initialize_system();

  run_iterations(1e-10, 10000, false);
}
} // end namespace MyNSE

int main() {
  try {
    using namespace dealii;
    using namespace MyNSE;

    const unsigned int degree = 1;
    const double Re = 1.0;
    const double rho = 2240.4;
    const double alpha = 1.0;
    const int m = 2;

    std::ofstream out_file("Refinement_Results.csv", std::ios_base::app);
    out_file.precision(10);
    out_file.seekp(0, std::ios::end);
    if (out_file.tellp() == 0) {
      out_file << "rho,m,N,Algorithm,Iterations,CPUTime,"
               << "L2_velocity,H1_velocity,L2_pressure,L2_divergence\n";
    }

    // from N=2 to N=8, run the simulation and write results to the file
    for (unsigned int N = 2; N <= 8; ++N) {
      std::cout << "\n======================================================\n";
      std::cout << "Running for Refinement Level N = " << N << " (m = " << m
                << ", rho = " << rho << ")\n";
      std::cout << "======================================================\n";

      Timer timer;
      AccuracyTest<2> flow(degree, Re, rho, alpha, m);

      flow.run(N);

      timer.stop();

      std::string algo_name =
          (m == 0) ? "IAH" : "AA-IAH(m=" + std::to_string(m) + ")";

      // Each (N, m) row now also stores the final-state errors.
      out_file << rho << "," << m << "," << N << "," << algo_name << ","
               << flow.get_final_iter() << "," << timer.cpu_time() << ","
               << flow.get_velocity_error_l2() << ","
               << flow.get_velocity_error_h1() << ","
               << flow.get_pressure_error_l2() << ","
               << flow.get_divergence_error_l2() << "\n";

      std::cout << "--> N = " << N
                << " completed! CPU Time: " << timer.cpu_time()
                << "s, Iterations: " << flow.get_final_iter() << std::endl;
    }
    out_file.close();

    std::cout << "\n******************************************************\n";
    std::cout << "All refinement levels have been processed and results "
                 "written to 'Refinement_Results.csv'"
              << std::endl;
    std::cout << "******************************************************\n\n";

  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
