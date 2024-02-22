// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Mark Sears (SNL)
------------------------------------------------------------------------- */

#include "temper.h"
#include "temper_vendi.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "random_park.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "library.h"
#include <unistd.h>

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define TEMPER_DEBUG 0

/* ---------------------------------------------------------------------- */

TemperVendi::TemperVendi(LAMMPS *lmp) : Command(lmp) {}

/* ---------------------------------------------------------------------- */

TemperVendi::~TemperVendi()
{
  MPI_Comm_free(&roots);
  if (ranswap) delete ranswap;
  delete ranboltz;
  delete[] set_temp;
  delete[] temp2world;
  delete[] world2temp;
  delete[] world2root;
}

/* ----------------------------------------------------------------------
   perform tempering with inter-world swaps
------------------------------------------------------------------------- */

void TemperVendi::command(int narg, char **arg)
{

  if (universe->nworlds == 1)
    error->universe_all(FLERR,"More than one processor partition required for temper command");
  if (domain->box_exist == 0)
    error->universe_all(FLERR,"Temper command before simulation box is defined");
  if (narg != 6 && narg != 7) error->universe_all(FLERR,"Illegal temper command");

  int nsteps = utils::inumeric(FLERR,arg[0],false,lmp);
  nevery = utils::inumeric(FLERR,arg[1],false,lmp);
  double temp = utils::numeric(FLERR,arg[2],false,lmp);

  // ignore temper command, if walltime limit was already reached

  if (timer->is_timeout()) return;

  whichfix = modify->get_fix_by_id(arg[3]);
  if (!whichfix)
    error->universe_all(FLERR,fmt::format("Tempering fix ID {} is not defined", arg[3]));

  seed_swap = utils::inumeric(FLERR,arg[4],false,lmp);
  seed_boltz = utils::inumeric(FLERR,arg[5],false,lmp);

  my_set_temp = universe->iworld;
  if (narg == 7) my_set_temp = utils::inumeric(FLERR,arg[6],false,lmp);
  if ((my_set_temp < 0) || (my_set_temp >= universe->nworlds))
    error->universe_one(FLERR,"Invalid temperature index value");

  // swap frequency must evenly divide total # of timesteps

  if (nevery <= 0)
    error->universe_all(FLERR,"Invalid frequency in temper command");
  nswaps = nsteps/nevery;
  if (nswaps*nevery != nsteps)
    error->universe_all(FLERR,"Non integer # of swaps in temper command");

  // fix style must be appropriate for temperature control, i.e. it needs
  // to provide a working Fix::reset_target() and must not change the volume.

  if ((!utils::strmatch(whichfix->style,"^nvt")) &&
      (!utils::strmatch(whichfix->style,"^langevin")) &&
      (!utils::strmatch(whichfix->style,"^gl[de]$")) &&
      (!utils::strmatch(whichfix->style,"^rigid/nvt")) &&
      (!utils::strmatch(whichfix->style,"^temp/")))
    error->universe_all(FLERR,"Tempering temperature fix is not supported");

  // setup for long tempering run

  update->whichflag = 1;
  timer->init_timeout();

  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  if (update->laststep < 0) error->all(FLERR,"Too many timesteps");

  lmp->init();

  // local storage

  me_universe = universe->me;
  MPI_Comm_rank(world,&me);
  nworlds = universe->nworlds;
  iworld = universe->iworld;
  boltz = force->boltz;

  std::string out = "======DEBUG=====\n";
  out += fmt::format("me_universe = {}, iworld = {}\n", me_universe, iworld);
  out += "======DEBUG=====\n";
  utils::logmesg(lmp,out);

  // pe_compute = ptr to thermo_pe compute
  // notify compute it will be called at first swap

  Compute *pe_compute = modify->get_compute_by_id("thermo_pe");
  if (!pe_compute) error->all(FLERR,"Tempering could not find thermo_pe compute");

  pe_compute->addstep(update->ntimestep + nevery);

  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(universe->uworld,color,0,&roots);

  // RNGs for swaps and Boltzmann test
  // warm up Boltzmann RNG

  if (seed_swap) ranswap = new RanPark(lmp,seed_swap);
  else ranswap = nullptr;
  ranboltz = new RanPark(lmp,seed_boltz + me_universe);
  for (int i = 0; i < 100; i++) ranboltz->uniform();

  // world2root[i] = global proc that is root proc of world i

  world2root = new int[nworlds];
  if (me == 0)
    MPI_Allgather(&me_universe,1,MPI_INT,world2root,1,MPI_INT,roots);
  MPI_Bcast(world2root,nworlds,MPI_INT,0,world);

  // create static list of set temperatures
  // allgather tempering arg "temp" across root procs
  // bcast from each root to other procs in world

  set_temp = new double[nworlds];
  if (me == 0) MPI_Allgather(&temp,1,MPI_DOUBLE,set_temp,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);

  // create world2temp only on root procs from my_set_temp
  // create temp2world on root procs from world2temp,
  //   then bcast to all procs within world

  world2temp = new int[nworlds];
  temp2world = new int[nworlds];
  if (me == 0) {
    MPI_Allgather(&my_set_temp,1,MPI_INT,world2temp,1,MPI_INT,roots);
    for (int i = 0; i < nworlds; i++) temp2world[world2temp[i]] = i;
  }
  MPI_Bcast(temp2world,nworlds,MPI_INT,0,world);

  // if restarting tempering, reset temp target of Fix to current my_set_temp

  if (narg == 7) {
    double new_temp = set_temp[my_set_temp];
    whichfix->reset_target(new_temp);
  }

  // setup tempering runs

  int i,which,partner,swap,partner_set_temp,partner_world;
  double pe,pe_partner,boltz_factor,new_temp;

  if (me_universe == 0 && universe->uscreen) {
    fprintf(universe->uscreen, "Setting up tempering ...\n");
  }

  update->integrate->setup(1);

  if (me_universe == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->uscreen," T%d",i);
      fprintf(universe->uscreen,"\n");
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->ulogfile," T%d",i);
      fprintf(universe->ulogfile,"\n");
    }
    print_status();
  }

  timer->init();
  timer->barrier_start();

  for (int iswap = 0; iswap < nswaps; iswap++) {

    // run for nevery timesteps

    timer->init_timeout();
    update->integrate->run(nevery);

    // check for timeout across all procs

    int my_timeout = 0;
    int any_timeout = 0;
    if (timer->is_timeout()) my_timeout = 1;
    MPI_Allreduce(&my_timeout, &any_timeout, 1, MPI_INT, MPI_SUM, universe->uworld);
    if (any_timeout) {
      timer->force_timeout();
      break;
    }

    // compute PE
    // notify compute it will be called at next swap

    pe = pe_compute->compute_scalar();
    pe_compute->addstep(update->ntimestep + nevery);

    // ------------- custom ----------------- //
    print_positions();
    // ------------- custom ----------------- //

  }



  timer->barrier_stop();

  update->integrate->cleanup();

  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/* ----------------------------------------------------------------------
   scale kinetic energy via velocities a la Sugita
------------------------------------------------------------------------- */

void TemperVendi::scale_velocities(int t_partner, int t_me)
{
  double sfactor = sqrt(set_temp[t_partner]/set_temp[t_me]);

  double **v = atom->v;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    v[i][0] = v[i][0]*sfactor;
    v[i][1] = v[i][1]*sfactor;
    v[i][2] = v[i][2]*sfactor;
  }
}

/* ----------------------------------------------------------------------
   proc 0 prints current tempering status
------------------------------------------------------------------------- */

void TemperVendi::print_status()
{
  std::string status = std::to_string(update->ntimestep);
  for (int i = 0; i < nworlds; i++)
    status += " " + std::to_string(world2temp[i]);

  status += "\n";

  if (universe->uscreen) fputs(status.c_str(), universe->uscreen);
  if (universe->ulogfile) {
    fputs(status.c_str(), universe->ulogfile);
    fflush(universe->ulogfile);
  }
}

void TemperVendi::print_positions()
{
  int natoms = lammps_get_natoms(lmp);
  double *x2 = (double *) malloc(nworlds*3*natoms*sizeof(double));
  double x3[nworlds][natoms][3];

  MPI_Allgather(atom->x, 3*natoms, MPI_DOUBLE, x3, 3*natoms, MPI_DOUBLE, roots);

  if (me_universe == 0) {
    std::string out = fmt::format("======POSITIONS=====\n");
    for (int i = 0; i < nworlds; i++) {
      out += fmt::format("======World {}=====\n", i);
      for (int j = 0; j < natoms; j++) {
        for (int k = 0; k < 3; k++) {
          //out += fmt::format("{}, ", x2[i*(natoms*3) + j*3 + k]);
          out += fmt::format("{}, ", x3[i][j][k]);
        }
        out += "\n";
      }
    }
    out += "======POSITIONS=====\n";

    utils::logmesg(lmp,out);
  }
}

//// Function to calculate the mean along a specified dimension
//void mean(double array[][3], int n_particles, double result[]) {
//  for (int k = 0; k < 3; ++k) {
//    for (int j = 0; j < n_particles; ++j) {
//      result[k] += array[j][k];
//    }
//
//    result[k] /= n_particles;
//  }
//}
//
//// Function to calculate the norm along a specified dimension
//double norm(double array[][3], int n_particles) {
//  double result = 0.0;
//
//  for (int j = 0; j < n_particles; ++j) {
//    for (int k = 0; k < 3; ++k) {
//      result += std::pow(array[j][k], 2);
//    }
//  }
//
//  return std::sqrt(result);
//}
//
//// Function to calculate the cross product
//void cross(double a[3], double b[3], double result[3]) {
//  result[0] = a[1] * b[2] - a[2] * b[1];
//  result[1] = a[2] * b[0] - a[0] * b[2];
//  result[2] = a[0] * b[1] - a[1] * b[0];
//}
//
//// Function to perform matrix multiplication
//void matmul(double a[][3], double b[][3], int m, int n, int p, double result[][3]) {
//  for (int i = 0; i < m; ++i) {
//    for (int j = 0; j < p; ++j) {
//      result[i][j] = 0.0;
//      for (int k = 0; k < n; ++k) {
//        result[i][j] += a[i][k] * b[k][j];
//      }
//    }
//  }
//}
//
//void getInvariant(double positions[][3][3], int n_replicas, int n_particles, double new_pos[][1][9]) {
//  for (int i = 0; i < n_replicas; ++i) {
//    double inp[1][3][3];
//    for (int j = 0; j < n_particles; ++j) {
//      for (int k = 0; k < 3; ++k) {
//        inp[0][j][k] = positions[i][j][k];
//      }
//    }
//
//    double center_of_mass[3] = {0.0};
//    mean(inp[0], n_particles, center_of_mass);
//
//    double centered[1][3][3];
//    for (int j = 0; j < n_particles; ++j) {
//      for (int k = 0; k < 3; ++k) {
//        centered[0][j][k] = inp[0][j][k] - center_of_mass[k];
//      }
//    }
//
//    double top[3];
//    for (int k = 0; k < 3; ++k) {
//      top[k] = centered[0][n_particles - 1][k];
//    }
//
//    double A[3];
//    double norm_top = norm(&top, 1);
//    for (int k = 0; k < 3; ++k) {
//      A[k] = top[k] / norm_top;
//    }
//
//    double B[3] = {1.0, 0.0, 0.0};
//
//    double dot = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
//
//    double cross_result[3];
//    cross(A, B, cross_result);
//
//    double cross_norm = norm(&cross_result, 1);
//
//    double zeros[3] = {0.0};
//    double ones[3] = {1.0, 1.0, 1.0};
//    double G1[2][3] = {{dot, -cross_norm, zeros[0]}, {zeros[1], zeros[2], ones[0]}};
//    double G2[2][3] = {{cross_norm, dot, zeros[1]}, {zeros[0], zeros[2], ones[1]}};
//    double G3[2][3] = {{zeros[2], zeros[0], ones[2]}, {zeros[0], zeros[1], ones[2]}};
//
//    double G[3][2][3] = {G1, G2, G3};
//
//    double u[3];
//    for (int k = 0; k < 3; ++k) {
//      u[k] = A[k];
//    }
//
//    double v[3];
//    for (int k = 0; k < 3; ++k) {
//      v[k] = (B[k] - (dot * A[k])) / norm(&(B[k] - (dot * A[k])), 1);
//    }
//
//    double w[3];
//    cross(B, A, w);
//
//    double F_inv[1][3][3] = {{u, v, w}};
//
//    double F[3][3];
//    matmul(&F_inv[0][0][0], &G[0][0][0], 3, 3, 2, &F[0][0]);
//
//    double rotated[1][3][3];
//    matmul(&centered[0][0][0], &F[0][0], 1, 3, 3, &rotated[0][0][0]);
//
//    for (int j = 0; j < 9; ++j) {
//      new_pos[i][0][j] = rotated[0][j / 3][j % 3];
//    }
//  }
//}