#ifndef __STARPU_CODELET_PARAMETERS_H__
#define __STARPU_CODELET_PARAMETERS_H__

#include <starpu.h>
#include "../StarPUUtils/FStarPUUtils.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

/* P2P */
static const char *p2p_cl_in_parameters_names[] = { "NbLeavesBlock", "SizeInterval", "NbParticlesGroup", "NbInteractions" };
static unsigned p2p_cl_in_combi1 [4] = { 0, 0, 1, 0 };
static unsigned p2p_cl_in_combi2 [4] = { 0, 0, 0, 1 };
static unsigned *p2p_cl_in_combinations[] = { p2p_cl_in_combi1, p2p_cl_in_combi2 };
static inline void p2p_cl_in_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int i;
  starpu_codelet_unpack_args(task->cl_arg,
	  	  	  	  &wrapperptr,
				  &i,
			       	  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
				  &parameters[3]);
}

/* P2P_out */
  static const char *p2p_cl_inout_parameters_names[] = { "NbLeavesBlock", "SizeInterval", "NbParticlesGroup", "iNbLeavesBlock", "iSizeInterval", "iNBParticlesGroup", "OutsideInteractionsSize", "NbDiff0", "NbDiff1", "NbInteractions" };
static unsigned p2p_cl_inout_combi1 [10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
static unsigned p2p_cl_inout_combi2 [10] = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
static unsigned p2p_cl_inout_combi3 [10] = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
static unsigned p2p_cl_inout_combi4 [10] = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
static unsigned p2p_cl_inout_combi5 [10] = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
static unsigned p2p_cl_inout_combi6 [10] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
static unsigned p2p_cl_inout_combi7 [10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
static unsigned *p2p_cl_inout_combinations[] = { p2p_cl_inout_combi1, p2p_cl_inout_combi2, p2p_cl_inout_combi3, p2p_cl_inout_combi4, p2p_cl_inout_combi5, p2p_cl_inout_combi6, p2p_cl_inout_combi7 };
static inline void p2p_cl_inout_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  std::vector<OutOfBlockInteraction>* outsideInteractions;
  int i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &outsideInteractions,
				  &i,
				  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3],
			     	  &parameters[4],
			     	  &parameters[5],
			       	  &parameters[6],
			     	  &parameters[7],
			     	  &parameters[8],
			          &parameters[9]);
}

/* P2M */
static const char *p2m_cl_parameters_names[] = { "NbCellsBlock", "SizeInterval", "NbParticlesGroup" };
static unsigned p2m_cl_combi1 [3] = { 0, 0, 1 };
static unsigned *p2m_cl_combinations[] = { p2m_cl_combi1 };
static inline void p2m_cl_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &i,
			       	  &parameters[0],
                                  &parameters[1],
				  &parameters[2]);
}
  
/* M2M */
static const char *m2m_cl_parameters_names[] = { "Lvl", "NbCellsBlock", "SizeInterval", "NbCellsBlockLvl+1", "SizeIntervalLvl+1", "NbDiff", "NbChildParent" };
static unsigned m2m_cl_combi1 [7] = { 0, 0, 0, 0, 0, 1, 0 };
static unsigned m2m_cl_combi2 [7] = { 0, 0, 0, 0, 1, 1, 0 };
static unsigned *m2m_cl_combinations[] = { m2m_cl_combi1, m2m_cl_combi2 };
static inline void m2m_cl_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int idxLevel, i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &idxLevel,
				  &i,
				  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3],
			     	  &parameters[4],
			     	  &parameters[5],
                                  &parameters[6]);
}
  
/* M2L */
static const char *m2l_cl_in_parameters_names[] = { "Lvl", "NbLeavesBlock", "SizeInterval", "NbM2LInteractions" };
static unsigned m2l_cl_in_combi1 [4] = { 0, 0, 0, 1 };
static unsigned *m2l_cl_in_combinations[] = { m2l_cl_in_combi1 };
static inline void m2l_cl_in_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int idxLevel, i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &idxLevel,
				  &i,
			     	  &parameters[0],
     			     	  &parameters[1],
     			     	  &parameters[2],
				  &parameters[3]);
}

/* M2L_out */
static const char *m2l_cl_inout_parameters_names[] = { "Lvl", "NbLeavesBlock", "SizeInterval", "iNbLeavesBlock", "iSizeInterval", "OutsideInteractionsSize", "NbDiff0", "NbDiff1" };
static unsigned m2l_cl_inout_combi1 [8] = { 0, 0, 0, 0, 0, 1, 0, 0 };
static unsigned m2l_cl_inout_combi2 [8] = { 0, 0, 0, 0, 0, 0, 1, 0 };
static unsigned *m2l_cl_inout_combinations[] = { m2l_cl_inout_combi1, m2l_cl_inout_combi2 };
static inline void m2l_cl_inout_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  const std::vector<OutOfBlockInteraction>* outsideInteractions;
  int idxLevel, i, m;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &idxLevel,
			          &outsideInteractions,
				  &i,
                                  &m,
				  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3],
			     	  &parameters[4],
			     	  &parameters[5],
			      	  &parameters[6],
                                  &parameters[7]);
}

/* L2L */
static const char *l2l_cl_parameters_names[] = { "Lvl", "NbCellsBlock", "SizeInterval", "NbCellsBlockLvl+1", "SizeIntervalLvl+1", "NbDiff", "NbChildParent" };
static unsigned l2l_cl_combi1 [7] = { 0, 0, 0, 0, 0, 1, 0 };
static unsigned *l2l_cl_combinations[] = { l2l_cl_combi1 };
static inline void l2l_cl_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int idxLevel, i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &idxLevel,
				  &i,
				  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3],
			     	  &parameters[4],
			     	  &parameters[5],
                                  &parameters[6]);
}

/* L2L_NOCOMMUTE */
static const char *l2l_cl_nocommute_parameters_names[] = { "Lvl", "NbCellsBlock", "SizeInterval", "NbCellsBlockLvl+1", "SizeIntervalLvl+1", "NbDiff", "NbChildParent" };
static unsigned l2l_cl_nocommute_combi1 [7] = { 0, 0, 0, 0, 0, 1, 0 };
static unsigned *l2l_cl_nocommute_combinations[] = { l2l_cl_nocommute_combi1 };
static inline void l2l_cl_nocommute_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int idxLevel, i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &idxLevel,
				  &i,
			     	  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3],
			     	  &parameters[4],
			     	  &parameters[5],
                                  &parameters[6]);
}

/* L2P */
static const char *l2p_cl_parameters_names[] = { "NbCellsBlock", "SizeInterval", "NbParticlesGroup" };
static unsigned l2p_cl_combi1 [3] = { 0, 0, 1 };
static unsigned *l2p_cl_combinations[] = { l2p_cl_combi1 };
static inline void l2p_cl_perf_func(struct starpu_task *task, double *parameters){
  FStarPUPtrInterface* wrapperptr;
  int i;
  starpu_codelet_unpack_args(task->cl_arg,
      	  	  	  	  &wrapperptr,
				  &i,
			      	  &parameters[0],
                                  &parameters[1],
			      	  &parameters[2],
			     	  &parameters[3]);
}
  
#ifdef __cplusplus
}
#endif

#endif /* __STARPU_CODELET_PARAMETERS_H__ */
