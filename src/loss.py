# loss.py

import torch
import math

def partial_coverage_penalty(coverage):
    gap = (0.5 - coverage).clamp(min=0.0)
    return 60.0 * gap.pow(2).mean()

def coverage_loss_simplified(coverage):
    gap = (1.0 - coverage).clamp(min=0.0)
    return 350.0 * gap.pow(2).mean()

def subset_sum_loss(probs, weights, target_sum):
    device = probs.device
    w_t = torch.tensor(weights, device=device)
    sum_ = torch.dot(probs, w_t)
    main_loss = (sum_ - target_sum) ** 2
    mid_val = (probs * (1.0 - probs)).mean()
    sep_factor = 20.0
    sep_loss = sep_factor * mid_val
    return main_loss + sep_loss

def hypermaxcut_loss(probs, incidence_matrix):
    device = probs.device
    n_edges = incidence_matrix.size(1)
    cut_value = torch.zeros((), device=device)

    for j in range(n_edges):
        if incidence_matrix.is_sparse:
            mask = (incidence_matrix.indices()[1] == j)
            node_idx = incidence_matrix.indices()[0][mask]
        else:
            col_j = incidence_matrix[:, j]
            node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
        if len(node_idx) == 0:
            continue
        p_vals = probs[node_idx]
        p_all_1 = p_vals.prod()
        p_all_0 = (1.0 - p_vals).prod()
        e_cut_prob = 1.0 - (p_all_1 + p_all_0)
        cut_value += e_cut_prob

    return -cut_value

def hypermultiwaycut_loss(partition_probs, incidence_matrix):
    device = partition_probs.device
    n_edges = incidence_matrix.size(1)
    cut_value = torch.zeros((), device=device)

    for j in range(n_edges):
        if incidence_matrix.is_sparse:
            mask = (incidence_matrix.indices()[1] == j)
            node_idx = incidence_matrix.indices()[0][mask]
        else:
            col_j = incidence_matrix[:, j]
            node_idx = (col_j > 0.5).nonzero(as_tuple=True)[0]
        if len(node_idx) == 0:
            continue
        pvals = partition_probs[node_idx]
        k = pvals.size(1)
        p_mono = torch.zeros((), device=device)
        for m in range(k):
            p_mono += pvals[:, m].prod()
        e_cut_prob = 1.0 - p_mono
        cut_value += e_cut_prob

    return -cut_value

def combined_loss(probs, incidence_matrix, epoch, total_epochs, 
                  in_phase2=False,
                  problem_type="set_cover",
                  extra_info=None):
    if problem_type == "set_cover":
        coverage = torch.matmul(incidence_matrix, probs)
        cov_loss = coverage_loss_simplified(coverage)
        part_loss = partial_coverage_penalty(coverage)
        
        # Improved set minimization penalty
        n_elem = incidence_matrix.size(0)
        n_subsets = incidence_matrix.size(1)
        
        # Use more realistic ideal density based on problem structure
        # Adaptive estimation based on problem density
        total_connections = incidence_matrix.sum().item()
        avg_set_size = total_connections / n_subsets if n_subsets > 0 else 1
        
        # For dense problems (large avg set size), need fewer sets
        # For sparse problems (small avg set size), need more sets
        if avg_set_size > n_elem * 0.1:  # Dense case
            estimated_optimal_sets = max(3, math.log(n_elem))
        else:  # Sparse case
            estimated_optimal_sets = max(math.log(n_elem), math.sqrt(n_elem) / 2)
            
        ideal_density = estimated_optimal_sets / n_subsets
        ideal_density = max(ideal_density, 0.005)  # Minimum 0.5%
        
        mean_p = probs.mean()
        progress = float(epoch) / (float(total_epochs) + 1e-9)
        
        # Stronger density penalty that increases with progress
        density_weight = 5.0 + 15.0 * progress  # Much stronger than before
        density_gap = (mean_p - ideal_density) / (ideal_density + 1e-9)
        density_loss = (density_gap ** 2) * density_weight
        
        # Separation loss to encourage binary decisions
        sep_factor = 10.0 + 30.0 * progress
        mid_val = (probs * (1 - probs)).mean()
        sep_loss = sep_factor * mid_val
        
        # Adaptive L1 penalty based on problem characteristics
        # Consider both size and density for optimal balance
        
        # Estimate expected solution size based on problem structure
        if avg_set_size > n_elem * 0.1:  # Very dense problems
            expected_sets = max(2, math.log(n_elem) * 0.5)
        elif avg_set_size > n_elem * 0.01:  # Moderately dense
            expected_sets = max(math.log(n_elem), math.sqrt(n_elem) / 4)
        else:  # Sparse problems
            expected_sets = max(math.sqrt(n_elem) / 2, n_elem * 0.1)
        
        # Scale L1 penalty to be proportional to expected solution size
        # This prevents over-penalization on small problems and under-penalization on large ones
        target_prob_sum = expected_sets / n_subsets
        l1_base = 0.01 * (350.0 / expected_sets)  # Scale relative to coverage loss
        l1_weight = l1_base * (1.0 + progress)  # Gentle increase during training
        
        # Only apply L1 penalty when we have reasonable coverage AND excess selection
        current_coverage = (coverage >= 1.0).float().mean().item()  # Fraction of elements covered
        
        if current_coverage > 0.8:  # Only minimize sets when coverage is good
            excess_selection = torch.clamp(probs.sum() - target_prob_sum * n_subsets, min=0)
            l1_loss = l1_weight * excess_selection
        else:
            l1_loss = torch.tensor(0.0, device=probs.device)  # No set minimization when coverage is poor
        
        total_loss = cov_loss + part_loss + density_loss + sep_loss + l1_loss
        
        # Phase 2: Additional set minimization (but not overwhelming)
        if in_phase2:
            phase2_penalty = 0.1 if n_elem < 1000 else 0.05  # Moderate additional penalty
            total_loss += phase2_penalty * probs.sum()
        
        return total_loss

    elif problem_type == "hitting_set":
        coverage = torch.matmul(incidence_matrix, probs)
        cov_loss = coverage_loss_simplified(coverage)
        part_loss = partial_coverage_penalty(coverage)
        l1_penalty = 0.05 * probs.sum()
        progress = float(epoch) / (float(total_epochs) + 1e-9)
        sep_factor = 10.0 + 30.0 * progress
        mid_val = (probs * (1 - probs)).mean()
        sep_loss = sep_factor * mid_val
        total_loss = cov_loss + part_loss + l1_penalty + sep_loss
        if in_phase2:
            total_loss += 0.0002 * probs.sum()
        return total_loss

    elif problem_type == "subset_sum":
        if extra_info is None:
            raise ValueError("subset_sum requires extra_info=(weights, target_sum).")
        (weights, target_sum) = extra_info
        return subset_sum_loss(probs, weights, target_sum)

    elif problem_type == "hypermaxcut":
        return hypermaxcut_loss(probs, incidence_matrix)

    elif problem_type == "hypermultiwaycut":
        return hypermultiwaycut_loss(probs, incidence_matrix)

    else:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
