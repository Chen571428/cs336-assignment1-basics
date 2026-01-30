use std::sync::Mutex;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use rustc_hash::FxHashMap;

use pyo3::prelude::*;
type HashMap<K, V> = FxHashMap<K, V>;

#[pyclass(module = "rust_utils")]
struct RustBPE {
    merges: HashMap<(u16, u16), (u16, u16)>,
    cache: Mutex<HashMap<Vec<u16>, Vec<u16>>>,
}

#[pymethods]
impl RustBPE {
    #[new]
    fn new(merges: HashMap<(u16, u16), (u16,u16)>) -> Self {
        RustBPE { 
            merges,
            cache: Mutex::new(HashMap::default())
        }
    }

    fn merge(&self, ids: Vec<u16>) -> Vec<u16> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached_result) = cache.get(&ids) {
                return cached_result.clone();
            }
        }
        
        let mut parts: Vec<(u16, usize, usize)> = ids.iter().enumerate()
            .map(|(i, &id)| (id, if i > 0 { i - 1 } else { usize::MAX }, if i < ids.len() - 1 { i + 1 } else { usize::MAX }))
            .collect();

        // (rank, start_index)
        let mut queue: BinaryHeap<Reverse<(u16, usize)>> = BinaryHeap::new();

        // Initial population
        for i in 0..parts.len().saturating_sub(1) {
            let pair = (parts[i].0, parts[i+1].0);
            if let Some(&(rank, _)) = self.merges.get(&pair) {
                queue.push(Reverse((rank, i)));
            }
        }

        while let Some(Reverse((rank, idx))) = queue.pop() {
            // Check if valid
            if parts[idx].0 == u16::MAX || parts[idx].2 == usize::MAX {
                continue; // Already merged or invalid
            }
            
            let next_idx = parts[idx].2;
            let pair = (parts[idx].0, parts[next_idx].0);
            
            // Validate that this is the current best merge for this pair
            // Because of lazy deletion, we might pull old entries
            if let Some(&(current_rank, new_id)) = self.merges.get(&pair) {
                if current_rank != rank {
                    continue;
                }

                // Apply merge
                parts[idx].0 = new_id;
                
                // Update links to remove next_idx
                let next_next_idx = parts[next_idx].2;
                parts[idx].2 = next_next_idx;
                if next_next_idx != usize::MAX {
                    parts[next_next_idx].1 = idx;
                }
                
                // Mark next_idx as removed
                parts[next_idx].0 = u16::MAX; 

                // Add new potential merges
                // 1. (prev, new)
                let prev_idx = parts[idx].1;
                if prev_idx != usize::MAX {
                    let prev_pair = (parts[prev_idx].0, parts[idx].0);
                    if let Some(&(r, _)) = self.merges.get(&prev_pair) {
                        queue.push(Reverse((r, prev_idx)));
                    }
                }

                // 2. (new, next)
                if next_next_idx != usize::MAX {
                    let next_pair = (parts[idx].0, parts[next_next_idx].0);
                    if let Some(&(r, _)) = self.merges.get(&next_pair) {
                        queue.push(Reverse((r, idx)));
                    }
                }
            }
        }

        // Collect result
        let mut result = Vec::with_capacity(ids.len());
        let mut curr = 0;
        while curr < parts.len() {
             if parts[curr].0 != u16::MAX {
                 result.push(parts[curr].0);
                 curr = parts[curr].2;
                 if curr == usize::MAX { break; }
             } else {
                 // Should not happen if following the chain, but safe fallback
                 curr += 1;
             }
        }

        {
             let mut cache = self.cache.lock().unwrap();
             cache.insert(ids, result.clone());
        }

        result
    }
}

#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPE>()?; 
    Ok(())
}