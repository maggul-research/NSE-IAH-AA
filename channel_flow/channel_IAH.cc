/* ---------------------------------------------------------------------
 * Copyright (C) 2025
 * Authored by Mustafa Aggul and Sinan Ergen
 * Hacettepe University && Southern Methodist University
 * Hacettepe University && Balikesir University
 *
 * This is the implementation file for the qualitative testing of
 * the improved Arrow Hurwicz Method for Navier Stokes Equation.
 * (CHANNEL FLOW OVER A FULL STEP)
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

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace MyNSE {
using namespace dealii;
using namespace std;

struct RunParameters {
  double Re = 100.0;
  double rho = 100.0;
  double alpha = 100.0;
  unsigned int refinements = 4;
  double tolerance = 1e-6;
  unsigned int max_iterations = 10000;
  bool show_help = false;
};

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --re VALUE              Reynolds number (default: 100)\n"
      << "  --rho VALUE             IAH parameter rho (default: 100)\n"
      << "  --alpha VALUE           pressure parameter alpha (default: 100)\n"
      << "  --refinements N         global refinement level (default: 4)\n"
      << "  --tolerance VALUE       stopping tolerance (default: 1e-6)\n"
      << "  --max-iterations N      iteration limit (default: 10000)\n"
      << "  -h, --help              show this message\n";
}

RunParameters parse_arguments(const int argc, char **argv) {
  RunParameters parameters;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto require_value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing value for " + argument);
      return argv[++i];
    };

    if (argument == "--re")
      parameters.Re = std::stod(require_value());
    else if (argument == "--rho")
      parameters.rho = std::stod(require_value());
    else if (argument == "--alpha")
      parameters.alpha = std::stod(require_value());
    else if (argument == "--refinements") {
      const int value = std::stoi(require_value());
      if (value < 0)
        throw std::invalid_argument("--refinements must be nonnegative");
      parameters.refinements = static_cast<unsigned int>(value);
    } else if (argument == "--tolerance")
      parameters.tolerance = std::stod(require_value());
    else if (argument == "--max-iterations") {
      const int value = std::stoi(require_value());
      if (value <= 0)
        throw std::invalid_argument("--max-iterations must be positive");
      parameters.max_iterations = static_cast<unsigned int>(value);
    } else if (argument == "-h" || argument == "--help")
      parameters.show_help = true;
    else
      throw std::invalid_argument("Unknown option: " + argument);
  }

  if (!std::isfinite(parameters.Re) || !std::isfinite(parameters.rho) ||
      !std::isfinite(parameters.alpha) ||
      !std::isfinite(parameters.tolerance) || parameters.Re <= 0.0 ||
      parameters.rho <= 0.0 || parameters.alpha <= 0.0 ||
      parameters.tolerance <= 0.0 || parameters.max_iterations == 0)
    throw std::invalid_argument(
        "Re, rho, alpha, tolerance, and max-iterations must be positive.");

  return parameters;
}

void ensure_csv_header(const std::string &filename, const std::string &header) {
  std::ifstream input(filename);
  std::string existing_header;
  if (!input.good() || !std::getline(input, existing_header) ||
      existing_header.empty()) {
    std::ofstream output(filename, std::ios::trunc);
    output << header << '\n';
  } else if (existing_header != header)
    throw std::runtime_error("Incompatible CSV header in " + filename);
}

template <int dim> class ChannelFlow {
public:
  ChannelFlow(const unsigned int degree, const double Re, const double rho,
              const double alpha);
  unsigned int run(const unsigned int n_refinements, const double tolerance,
                   const unsigned int max_iteration);

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
  unsigned int run_iterations(const unsigned int n_refinements,
                              const double tolerance,
                              const unsigned int max_iteration,
                              const bool output_result);

  const unsigned int degree;
  const double Re;
  const double rho;
  const double alpha;

  const unsigned int solver_type;

  vector<types::global_dof_index> dofs_per_block;

  Triangulation<dim> triangulation;
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  BlockSparsityPattern sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> old_solution;
  BlockVector<double> solution;
  BlockVector<double> system_rhs;
  BlockVector<double> residue_vector;
};

/* RHS for Channel Flow is exactly zero */
template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
};

template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &values) const {
  values(0) = 0.0;
  values(1) = 0.0;
}

/* Parabolic Inlet Profile (Poiseuille) for the Left Boundary */
template <int dim> class InletBoundaryValues : public Function<dim> {
public:
  InletBoundaryValues() : Function<dim>(dim + 1) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override {
    double y = p[1];
    values = 0;
    values(0) = (y * (10.0 - y)) / 25.0;
    values(1) = 0.0;
  }
};

template <int dim>
ChannelFlow<dim>::ChannelFlow(const unsigned int degree, const double Re,
                              const double rho, const double alpha)
    : degree(degree), Re(Re), rho(rho), alpha(alpha),
      solver_type(1) /* GMRES:0 and UMFPACK:1 */
      ,
      triangulation(Triangulation<dim>::maximum_smoothing),
      fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)

      ,
      dof_handler(triangulation) {}

template <int dim>
void ChannelFlow<dim>::mesh_jobs(const unsigned int n_refinements) {

  Triangulation<dim> tria1, tria3, tria4, tria5, tria6;

  std::vector<unsigned int> rep1 = {5, 1};
  GridGenerator::subdivided_hyper_rectangle(tria1, rep1, Point<dim>(0.0, 0.0),
                                            Point<dim>(5.0, 1.0));

  std::vector<unsigned int> rep4 = {5, 9};
  GridGenerator::subdivided_hyper_rectangle(tria4, rep4, Point<dim>(0.0, 1.0),
                                            Point<dim>(5.0, 10.0));

  Triangulation<dim> left_col;
  GridGenerator::merge_triangulations(tria1, tria4, left_col);

  std::vector<unsigned int> rep3 = {24, 1};
  GridGenerator::subdivided_hyper_rectangle(tria3, rep3, Point<dim>(6.0, 0.0),
                                            Point<dim>(30.0, 1.0));

  std::vector<unsigned int> rep6 = {24, 9};
  GridGenerator::subdivided_hyper_rectangle(tria6, rep6, Point<dim>(6.0, 1.0),
                                            Point<dim>(30.0, 10.0));

  Triangulation<dim> right_col;
  GridGenerator::merge_triangulations(tria3, tria6, right_col);

  std::vector<unsigned int> rep5 = {1, 9};
  GridGenerator::subdivided_hyper_rectangle(tria5, rep5, Point<dim>(5.0, 1.0),
                                            Point<dim>(6.0, 10.0));

  Triangulation<dim> left_and_mid;
  GridGenerator::merge_triangulations(left_col, tria5, left_and_mid);

  GridGenerator::merge_triangulations(left_and_mid, right_col, triangulation);

  triangulation.refine_global(n_refinements);

  // Assign boundary IDs
  for (const auto &cell : triangulation.active_cell_iterators()) {
    if (cell->at_boundary()) {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        if (cell->face(f)->at_boundary()) {
          const auto center = cell->face(f)->center();
          if (std::abs(center[0] - 0.0) < 1e-6)
            cell->face(f)->set_boundary_id(1);
          else if (std::abs(center[0] - 30.0) < 1e-6)
            cell->face(f)->set_boundary_id(2);
          else
            cell->face(f)->set_boundary_id(0);
        }
      }
    }
  }

  std::cout << "\n--- MESH Information ---" << std::endl;
  std::cout << "  Total Cells Count: " << triangulation.n_active_cells()
            << std::endl;
  std::cout << "---------------------------\n" << std::endl;
}

template <int dim> void ChannelFlow<dim>::setup_dofs() {
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

    VectorTools::interpolate_boundary_values(
        dof_handler, 0, Functions::ZeroFunction<dim>(dim + 1), constraints,
        fe.component_mask(velocities));

    VectorTools::interpolate_boundary_values(
        dof_handler, 1, InletBoundaryValues<dim>(), constraints,
        fe.component_mask(velocities));
  }
  constraints.close();

  cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
       << dof_u << '+' << dof_p << ')' << endl;
}

template <int dim> void ChannelFlow<dim>::initialize_system() {
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit(sparsity_pattern);
  old_solution.reinit(dofs_per_block);
  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
  residue_vector.reinit(dofs_per_block);

  solution = 0;
}

template <int dim> void ChannelFlow<dim>::assemble(int assemble_id) {
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

template <int dim> void ChannelFlow<dim>::solve(int solve_id) {
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
unsigned int ChannelFlow<dim>::run_iterations(const unsigned int n_refinements,
                                              const double tolerance,
                                              const unsigned int max_iteration,
                                              const bool output_result) {

  unsigned int th_iteration = 0;
  double current_residue = 1000.0;
  bool first_step = true;

  while ((first_step || current_residue > tolerance) &&
         th_iteration < max_iteration) {
    old_solution = solution;

    assemble(0);
    solve(0);

    assemble(1);
    solve(1);

    first_step = false;

    residue_vector = solution;
    residue_vector -= old_solution;

    current_residue =
        max(residue_vector.block(0).l2_norm() / solution.block(0).l2_norm(),
            residue_vector.block(1).l2_norm() / solution.block(1).l2_norm());

    cout << "******************************" << endl;
    cout << " The relative error of the current iteration = " << current_residue
         << " at " << " at " << th_iteration << "th iteration" << endl;
    cout << "******************************" << endl;

    ++th_iteration;

    std::string algo_name = "IAH";
    std::ofstream res_file("residue_history.csv", std::ios::app);
    res_file << Re << "," << rho << "," << alpha << ",0," << n_refinements
             << "," << algo_name << "," << th_iteration << ","
             << current_residue << "\n";
    res_file.close();
    /*if (output_result)
    output_results(th_iteration);*/
    if (current_residue > 100000.0) {
      th_iteration = 1000;
      break;
    }
  }
  output_results(th_iteration);
  return th_iteration;
}

template <int dim>
void ChannelFlow<dim>::output_results(const unsigned int output_index) const {
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.build_patches();

  ostringstream filename;
  filename << "Channel_Re_" << Re << "_rho_" << rho << "_m_0_solution_"
           << Utilities::int_to_string(output_index, 4) << ".vtk";

  ofstream output(filename.str().c_str());
  data_out.write_vtk(output);
}

template <int dim>
unsigned int ChannelFlow<dim>::run(const unsigned int n_refinements,
                                   const double tolerance,
                                   const unsigned int max_iteration) {
  mesh_jobs(n_refinements);
  setup_dofs();
  initialize_system();

  return run_iterations(n_refinements, tolerance, max_iteration, true);
}
} // end namespace MyNSE

int main(int argc, char **argv) {
  try {
    using namespace dealii;
    using namespace MyNSE;

    const RunParameters parameters = parse_arguments(argc, argv);
    if (parameters.show_help) {
      print_usage(argv[0]);
      return 0;
    }

    const unsigned int degree = 1;
    ensure_csv_header("performance_summary.csv",
                      "Re,rho,alpha,m,N,Algorithm,TotalIterations,CPUTime");
    ensure_csv_header("residue_history.csv",
                      "Re,rho,alpha,m,N,Algorithm,Iteration,RelativeError");

    std::cout << "\n======================================================\n";
    std::cout << "Running IAH for Channel Flow: Re = " << parameters.Re
              << ", rho = " << parameters.rho
              << ", alpha = " << parameters.alpha
              << " (m = 0, N = " << parameters.refinements << ")\n";
    std::cout << "======================================================\n";

    Timer timer;
    timer.start();

    ChannelFlow<2> flow(degree, parameters.Re, parameters.rho,
                        parameters.alpha);

    const unsigned int total_iters =
        flow.run(parameters.refinements, parameters.tolerance,
                 parameters.max_iterations);

    timer.stop();
    const double cpu_time = timer.cpu_time();

    std::ofstream p_file("performance_summary.csv", std::ios::app);
    p_file << parameters.Re << "," << parameters.rho << "," << parameters.alpha
           << ",0," << parameters.refinements << ",IAH," << total_iters << ","
           << cpu_time << "\n";
    p_file.close();

    std::cout << "--> Re = " << parameters.Re << " | rho = " << parameters.rho
              << " completed! CPU Time: " << cpu_time
              << "s, Iterations: " << total_iters << std::endl;

    std::cout << "\n******************************************************\n";
    std::cout << "IAH completed. Results appended to the CSV files."
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
