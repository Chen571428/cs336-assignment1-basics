use std::sync::{Mutex, OnceLock};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use rustc_hash::FxHashMap;
use fancy_regex::Regex;

use pyo3::prelude::*;
type HashMap<K, V> = FxHashMap<K, V>;

static PREPATTERN: OnceLock<Regex> = OnceLock::new();

#[pyclass(module = "rust_utils")]
struct RustBPE {
    merges: HashMap<(u16, u16), (u16, u16)>,
    byte_encoder: [u16; 256],
    cache: Mutex<HashMap<Vec<u16>, Vec<u16>>>,
}

#[pymethods]
impl RustBPE {
    #[new]
    fn new(merges: HashMap<(u16, u16), (u16,u16)>,byte_encoder: [u16; 256]) -> Self {

        RustBPE { 
            merges,
            byte_encoder,
            cache: Mutex::new(HashMap::default())
        }
    }

    fn merge_heap(&self, ids: Vec<u16>) -> Vec<u16> {
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
    fn merge(&self, ids: Vec<u16>) -> Vec<u16> {
        if ids.len() < 2 {
            return ids;
        }
        if ids.len() > 24 {
            return self.merge_heap(ids);
        }

        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached_result) = cache.get(&ids) {
                return cached_result.clone();
            }
        }
        let mut working_ids = ids.clone();
        loop {
            let mut min_rank = u16::MAX;
            let mut best_idx = None;
            let mut best_new_id = 0;

            if working_ids.len() < 2 {
                break;
            }

            for i in 0..working_ids.len() - 1 {
                let pair = (working_ids[i], working_ids[i+1]);
                if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_idx = Some(i);
                        best_new_id = new_id;
                    }
                }
            }

            match best_idx {
                Some(idx) => {
                    working_ids[idx] = best_new_id;
                    working_ids.remove(idx + 1);
                },
                None => break,
            }
        }

        {
             let mut cache = self.cache.lock().unwrap();
             cache.insert(ids, working_ids.clone());
        }

        working_ids
    }

    fn encode_single_text(&self, s: &str) -> Vec<u16>{
        let re = PREPATTERN.get_or_init(||{
            Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap()
        });

        let mut encoded_str: Vec<u16> = Vec::new();
        
        for mat in re.find_iter(s) {
            let bytes = mat.unwrap().as_str().as_bytes();
            let byte_ids: Vec<u16> = bytes.iter().map(|&b| self.byte_encoder[b as usize]).collect();
            let merged = self.merge(byte_ids);
            encoded_str.extend(merged);
        }

        encoded_str
    }
}

#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPE>()?; 
    Ok(())
}