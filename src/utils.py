import torch
import math

def batch_coverage_check(solution, incidence_matrix, batch_size=32768):
    """
    For set_cover/hitting_set => fraction of rows with sum >= 1.
    Batches to avoid memory issues for large problems.
    """
    device = incidence_matrix.device
    n_rows = incidence_matrix.size(0)
    covered_count = 0

    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        if incidence_matrix.is_sparse:
            mask = (incidence_matrix.indices()[0] >= start_idx) & (incidence_matrix.indices()[0] < end_idx)
            indices = incidence_matrix.indices()[:, mask]
            indices_block = indices.clone()
            indices_block[0] = indices_block[0] - start_idx
            vals = incidence_matrix.values()[mask]
            block = torch.sparse_coo_tensor(
                indices_block, vals,
                size=(end_idx - start_idx, incidence_matrix.size(1)),
                device=device
            ).coalesce()
            coverage_sub = torch.sparse.mm(block, solution.unsqueeze(1)).squeeze(1)
        else:
            block = incidence_matrix[start_idx:end_idx]
            coverage_sub = block.matmul(solution.unsqueeze(1)).squeeze(1)

        covered_count += (coverage_sub >= 1.0).sum().item()

    uncovered_count = n_rows - covered_count
    coverage_ratio = covered_count / float(n_rows) if n_rows > 0 else 1.0
    return coverage_ratio, uncovered_count


def optimize_solution_size(solution, incidence_matrix, max_passes=2, use_int=True, problem_type="set_cover"):
    """
    Greedy pass: remove any selected item if coverage remains >=1 in all rows.
    For set_cover/hitting_set => reduce solution size.
    """
    sol = solution.clone()
    if incidence_matrix.is_sparse:
        mat_dense = incidence_matrix.to_dense()
    else:
        mat_dense = incidence_matrix

    cvec = mat_dense @ sol

    for _pass in range(max_passes):
        chosen = (sol > 0.5).nonzero(as_tuple=True)[0]
        changed = False
        for idx in chosen:
            if sol[idx] < 0.5:
                continue
            test_sol = sol.clone()
            test_sol[idx] = 0.0
            new_cvec = cvec - mat_dense[:, idx]
            if torch.all(new_cvec >= 1.0):
                sol[idx] = 0.0
                cvec = new_cvec
                changed = True
        if not changed:
            break

    return sol


def integrated_post_process_set_cover(best_probs, incidence_matrix):
    """
    Coverage-based greedy approach for set cover, using best_probs as tie-break,
    then local removal pass.
    """
    device = best_probs.device
    n_elem = incidence_matrix.size(0)
    n_sub = incidence_matrix.size(1)

    subset_to_elems = [[] for _ in range(n_sub)]
    elem_to_subsets = [[] for _ in range(n_elem)]

    if incidence_matrix.is_sparse:
        rows = incidence_matrix.indices()[0]
        cols = incidence_matrix.indices()[1]
        for i in range(len(rows)):
            e = rows[i].item()
            s = cols[i].item()
            subset_to_elems[s].append(e)
            elem_to_subsets[e].append(s)
    else:
        mat_bool = (incidence_matrix > 0.5)
        for s in range(n_sub):
            col_s = mat_bool[:, s]
            e_idx = col_s.nonzero(as_tuple=True)[0]
            subset_to_elems[s] = e_idx.tolist()
            for e in e_idx:
                elem_to_subsets[e].append(s)

    coverage_count = torch.zeros(n_sub, dtype=torch.float32, device=device)
    for s in range(n_sub):
        coverage_count[s] = len(subset_to_elems[s])

    chosen = torch.zeros(n_sub, dtype=torch.float32, device=device)
    covered = torch.zeros(n_elem, dtype=torch.bool, device=device)
    uncovered_count = n_elem

    alpha = 0.1
    while uncovered_count > 0:
        score = coverage_count + alpha * best_probs
        score[chosen > 0.5] = -1e9
        best_s = torch.argmax(score).item()
        if score[best_s] <= 0:
            break
        chosen[best_s] = 1
        newly_covered = subset_to_elems[best_s]
        for e in newly_covered:
            if not covered[e]:
                covered[e] = True
                uncovered_count -= 1
                for s2 in elem_to_subsets[e]:
                    coverage_count[s2] -= 1
                    if coverage_count[s2] < 0:
                        coverage_count[s2] = 0
        if uncovered_count <= 0:
            break

    from src.utils import optimize_solution_size
    chosen = optimize_solution_size(chosen, incidence_matrix, max_passes=2, problem_type="set_cover")
    return chosen


def integrated_post_process_hitting_set(best_probs, incidence_matrix):
    """
    Coverage-based greedy approach for hitting set, plus local removal pass.
    """
    device = best_probs.device
    n_sets = incidence_matrix.size(0)
    n_elem = incidence_matrix.size(1)
    chosen = torch.zeros(n_elem, device=device)
    uncovered = torch.ones(n_sets, dtype=torch.bool, device=device)

    if incidence_matrix.is_sparse:
        col_sums = torch.sparse.sum(incidence_matrix, dim=0)
        if col_sums.is_sparse:
            col_sums = col_sums.to_dense()
    else:
        col_sums = incidence_matrix.sum(dim=0)

    while torch.any(uncovered):
        if incidence_matrix.is_sparse:
            cover_score = torch.zeros(n_elem, device=device)
            rows = incidence_matrix.indices()[0]
            cols = incidence_matrix.indices()[1]
            mask = uncovered[rows]
            rel_cols = cols[mask]
            cover_score.index_add_(0, rel_cols, torch.ones_like(rel_cols, dtype=torch.float32))
        else:
            cover_score = incidence_matrix[uncovered].sum(dim=0)
        eff = cover_score / (col_sums + 1e-6)
        combined = 0.6 * cover_score + 0.4 * best_probs * eff
        combined[chosen > 0.5] = -1e9
        best_idx = torch.argmax(combined).item()
        if combined[best_idx] <= 0:
            break
        chosen[best_idx] = 1
        if incidence_matrix.is_sparse:
            mask_ = (incidence_matrix.indices()[1] == best_idx)
            c_rows = incidence_matrix.indices()[0][mask_]
            uncovered[c_rows] = False
        else:
            newly_cov = incidence_matrix[:, best_idx] > 0
            uncovered[newly_cov] = False

    from src.utils import optimize_solution_size
    chosen = optimize_solution_size(chosen, incidence_matrix, max_passes=2, problem_type="hitting_set")
    return chosen


def post_process_solution_hypermaxcut_fast(raw_sol, incidence_matrix, max_passes=2):
    """
    Local flipping approach for hypermaxcut: flip node side if #cut edges improves.
    """
    device = raw_sol.device
    sol = raw_sol.clone()
    n_nodes = sol.size(0)
    n_edges = incidence_matrix.size(1)

    node_to_edges = [[] for _ in range(n_nodes)]
    if incidence_matrix.is_sparse:
        rows = incidence_matrix.indices()[0]
        cols = incidence_matrix.indices()[1]
        for i in range(len(rows)):
            i_node = rows[i].item()
            e_id = cols[i].item()
            node_to_edges[i_node].append(e_id)
    else:
        mat_bool = (incidence_matrix > 0.5)
        for j in range(n_edges):
            col_j = mat_bool[:, j]
            node_idx = col_j.nonzero(as_tuple=True)[0]
            for i_node in node_idx:
                node_to_edges[i_node.item()].append(j)

    if incidence_matrix.is_sparse:
        size_e = torch.sparse.sum(incidence_matrix, dim=0).to_dense().long()
    else:
        size_e = (incidence_matrix > 0.5).sum(dim=0).long()

    count_ones = torch.zeros(n_edges, device=device, dtype=torch.long)
    for i in range(n_nodes):
        if sol[i] > 0.5:
            for e in node_to_edges[i]:
                count_ones[e] += 1

    cut_array = torch.zeros(n_edges, device=device, dtype=torch.bool)
    for e in range(n_edges):
        cut_array[e] = (count_ones[e] > 0) and (count_ones[e] < size_e[e])

    for _ in range(max_passes):
        changed = False
        order_nodes = torch.randperm(n_nodes, device=device)
        for i in order_nodes:
            old_val = sol[i]
            new_val = 1.0 - old_val
            touched_edges = node_to_edges[i.item()]
            old_cut_local = sum(cut_array[e] for e in touched_edges)
            new_cut_local = 0
            if old_val > 0.5:
                for e in touched_edges:
                    new_count = count_ones[e] - 1
                    if new_count > 0 and new_count < size_e[e]:
                        new_cut_local += 1
            else:
                for e in touched_edges:
                    new_count = count_ones[e] + 1
                    if new_count > 0 and new_count < size_e[e]:
                        new_cut_local += 1
            net_gain = new_cut_local - old_cut_local
            if net_gain > 0:
                sol[i] = new_val
                if old_val > 0.5:
                    for e in touched_edges:
                        count_ones[e] -= 1
                else:
                    for e in touched_edges:
                        count_ones[e] += 1
                for e in touched_edges:
                    c = count_ones[e]
                    cut_array[e] = (c > 0) and (c < size_e[e])
                changed = True
        if not changed:
            break

    return sol


def post_process_solution_hypermaxcut(raw_sol, incidence_matrix, max_passes=2):
    return post_process_solution_hypermaxcut_fast(raw_sol, incidence_matrix, max_passes)


def post_process_solution_hypermultiwaycut(partition_probs, incidence_matrix, max_passes=2):
    """
    Local flipping approach for hypermultiwaycut: flip node's partition if #cut edges improves.
    """
    device = partition_probs.device
    n_nodes = partition_probs.size(0)
    k = partition_probs.size(1)
    n_edges = incidence_matrix.size(1)

    assignments = torch.argmax(partition_probs, dim=1)

    node_to_edges = [[] for _ in range(n_nodes)]
    if incidence_matrix.is_sparse:
        rows = incidence_matrix.indices()[0]
        cols = incidence_matrix.indices()[1]
        for idx in range(len(rows)):
            i_node = rows[idx].item()
            e_id = cols[idx].item()
            node_to_edges[i_node].append(e_id)
    else:
        mat_bool = (incidence_matrix > 0.5)
        for j in range(n_edges):
            col_j = mat_bool[:, j]
            node_idx = col_j.nonzero(as_tuple=True)[0]
            for i_node in node_idx:
                node_to_edges[i_node.item()].append(j)

    if incidence_matrix.is_sparse:
        size_e = torch.sparse.sum(incidence_matrix, dim=0).to_dense().long()
    else:
        size_e = (incidence_matrix > 0.5).sum(dim=0).long()

    partition_count = torch.zeros(n_edges, k, device=device, dtype=torch.long)
    for i in range(n_nodes):
        p = assignments[i].item()
        for e in node_to_edges[i]:
            partition_count[e, p] += 1

    def is_cut(e):
        counts = partition_count[e]
        return (counts > 0).sum().item() > 1

    cut_array = torch.zeros(n_edges, dtype=torch.bool, device=device)
    for e in range(n_edges):
        cut_array[e] = is_cut(e)

    for _ in range(max_passes):
        changed = False
        order_nodes = torch.randperm(n_nodes, device=device)
        for i in order_nodes:
            old_p = assignments[i].item()
            e_list = node_to_edges[i]
            old_cut_local = sum(cut_array[e] for e in e_list)
            best_new_p = old_p
            best_gain = 0
            for new_p in range(k):
                if new_p == old_p:
                    continue
                new_cut_local = 0
                for e in e_list:
                    temp_counts = partition_count[e].clone()
                    temp_counts[old_p] = max(temp_counts[old_p] - 1, 0)
                    temp_counts[new_p] = temp_counts[new_p] + 1
                    if (temp_counts > 0).sum().item() > 1:
                        new_cut_local += 1
                gain = new_cut_local - old_cut_local
                if gain > best_gain:
                    best_gain = gain
                    best_new_p = new_p
            if best_new_p != old_p:
                assignments[i] = best_new_p
                for e in e_list:
                    partition_count[e, old_p] = max(partition_count[e, old_p] - 1, 0)
                    partition_count[e, best_new_p] += 1
                    cut_array[e] = is_cut(e)
                changed = True
        if not changed:
            break

    return assignments


def check_problem_feasibility(incidence_matrix, problem_type="set_cover", extra_info=None):
    """
    Basic feasibility checks:
      - set_cover/hitting_set => no empty row
      - subset_sum => sum(weights) >= target
      - hypermaxcut/hypermultiwaycut => always feasible
    """
    if problem_type == "set_cover":
        if incidence_matrix.is_sparse:
            row_sum = torch.sparse.sum(incidence_matrix, dim=1)
            if row_sum.is_sparse:
                row_sum = row_sum.to_dense()
        else:
            row_sum = incidence_matrix.sum(dim=1)
        return bool(torch.all(row_sum > 0))

    elif problem_type == "hitting_set":
        if incidence_matrix.is_sparse:
            row_sum = torch.sparse.sum(incidence_matrix, dim=1)
            if row_sum.is_sparse:
                row_sum = row_sum.to_dense()
        else:
            row_sum = incidence_matrix.sum(dim=1)
        return bool(torch.all(row_sum > 0))

    elif problem_type == "subset_sum":
        if extra_info is not None:
            (weights, target_sum) = extra_info
            if sum(weights) < target_sum:
                print("Warning: sum(weights) < target => infeasible.")
                return False
        return True

    elif problem_type in ["hypermaxcut", "hypermultiwaycut"]:
        return True

    else:
        return True


def validate_solution(solution, incidence_matrix, problem_type="set_cover", extra_info=None):
    """
    Basic validation:
      - set_cover/hitting_set => coverage >=1 in all rows
      - subset_sum => no nans/inf
      - hypermaxcut/hypermultiwaycut => skip
    """
    if solution is None:
        return False
    if torch.isnan(solution).any() or torch.isinf(solution).any():
        return False

    if problem_type == "set_cover":
        if solution.dim() > 1:
            solution = solution.squeeze()
        if solution.size(0) != incidence_matrix.size(1):
            return False
        coverage, uncov = batch_coverage_check(solution, incidence_matrix)
        return (uncov == 0)

    elif problem_type == "hitting_set":
        if solution.dim() > 1:
            solution = solution.squeeze()
        if solution.size(0) != incidence_matrix.size(1):
            return False
        if incidence_matrix.is_sparse:
            coverage_vals = torch.sparse.mm(incidence_matrix, solution.unsqueeze(1)).squeeze(1)
        else:
            coverage_vals = incidence_matrix @ solution
        return bool(torch.all(coverage_vals >= 1.0))

    elif problem_type == "subset_sum":
        return True

    else:
        return True


def post_process_solution_subset_sum(best_probs, weights, target_sum):
    """
    Local BFS/fix-up for subset_sum to get sum closer to target.
    """
    device = best_probs.device
    n_items = best_probs.size(0)
    thr = 0.5
    raw_sol = (best_probs >= thr).float()

    w_t = torch.tensor(weights, device=device)
    raw_sum = float((raw_sol * w_t).sum().item())
    best_sol = raw_sol.clone()
    best_diff = abs(raw_sum - target_sum)

    if n_items <= 20:
        indices_sorted = sorted(range(n_items), key=lambda i: float(best_probs[i]), reverse=True)
        chosen = []
        best_found = (best_sol, raw_sum)

        def backtrack(idx, current_sum, used):
            nonlocal best_found
            if idx == len(indices_sorted):
                diff = abs(current_sum - target_sum)
                if diff < abs(best_found[1] - target_sum):
                    sol_vec = torch.zeros(n_items, device=device)
                    for i_ in used:
                        sol_vec[i_] = 1.0
                    best_found = (sol_vec, current_sum)
                return
            i_item = indices_sorted[idx]
            w_item = float(weights[i_item])
            used.append(i_item)
            backtrack(idx+1, current_sum + w_item, used)
            used.pop()
            backtrack(idx+1, current_sum, used)

        backtrack(0, 0.0, [])
        best_sol = best_found[0]
        best_diff = abs(best_found[1] - target_sum)
    else:
        sum_now = raw_sum
        for _ in range(3):
            if sum_now > target_sum:
                selected_idx = (best_sol > 0.5).nonzero(as_tuple=True)[0].tolist()
                selected_idx.sort(key=lambda i: float(best_probs[i]))
                for i in selected_idx:
                    new_sum = sum_now - float(weights[i])
                    if abs(new_sum - target_sum) < abs(sum_now - target_sum):
                        best_sol[i] = 0.0
                        sum_now = new_sum
            else:
                not_sel_idx = (best_sol < 0.5).nonzero(as_tuple=True)[0].tolist()
                not_sel_idx.sort(key=lambda i: float(best_probs[i]), reverse=True)
                for i in not_sel_idx:
                    new_sum = sum_now + float(weights[i])
                    if abs(new_sum - target_sum) < abs(sum_now - target_sum):
                        best_sol[i] = 1.0
                        sum_now = new_sum
        best_diff = abs(sum_now - target_sum)

    return best_sol, best_diff
