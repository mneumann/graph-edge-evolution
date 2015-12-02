#![feature(augmented_assignments)]
#![feature(op_assign_traits)]

#[macro_use]
extern crate log;

use std::fmt::Debug;
use std::collections::BTreeMap;
use std::ops::AddAssign;
use std::ops::Neg;

#[derive(Debug, Clone)]
pub enum EdgeOperation<W: Clone, N: Clone> {
    IncreaseWeight {
        weight: W,
    },
    DecreaseWeight {
        weight: W,
    },
    Duplicate {
        weight: W,
    },
    Split {
        weight: W,
    },
    Loop {
        weight: W,
    },
    Output {
        weight: W,
    },
    Merge {
        n: u32,
    },
    Next {
        n: u32,
    },
    Parent {
        n: u32,
    },
    SetNodeFunction {
        function: N,
    },
    Reverse,

    // Saves the state
    Save,

    // Restores a previously saved state
    Restore,
}

#[derive(Debug, Clone)]
struct Edge<W: Debug + Clone> {
    src: usize,
    dst: usize,
    weight: W,
}

impl<W:Debug+Clone+Default> Edge<W> {
    fn new(src: usize, dst: usize) -> Edge<W> {
        Edge {
            src: src,
            dst: dst,
            weight: W::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct Node<N: Clone + Default + Debug> {
    in_edges: Vec<usize>,
    out_edges: Vec<usize>,
    node_fn: N,
}

impl<N:Default+Clone+Debug> Node<N> {
    fn new() -> Node<N> {
        Node {
            in_edges: Vec::new(),
            out_edges: Vec::new(),
            node_fn: N::default(),
        }
    }
}

#[derive(Debug)]
struct State {
    from_node: usize,
    to_node: usize,
    edge: usize,
}

#[derive(Debug)]
pub struct GraphBuilder<W: Debug + Default + Clone, N: Debug + Default + Clone> {
    edges: BTreeMap<usize, Edge<W>>,
    nodes: Vec<Node<N>>,
    next_edge_id: usize,
    states: Vec<State>,
    current_from_node: usize,
    current_to_node: usize,
    current_edge: usize,
}

impl<W:Debug+Default+Clone+AddAssign<W>, N:Debug+Default+Clone> GraphBuilder<W, N> {
    pub fn new() -> GraphBuilder<W, N> {
        // we begin with an empty node and a virtual edge.

        GraphBuilder {
            edges: BTreeMap::new(),
            nodes: vec![Node::new()],
            next_edge_id: 1,
            states: Vec::new(),
            current_from_node: 0,
            current_to_node: 0,
            current_edge: 0, /* this points to a non-existing edge by purpose! */
        }
    }

    /// Saves the current state to the internal state stack.
    pub fn save(&mut self) {
        let state = self.get_state();
        self.states.push(state);
    }

    /// Restore a previously saved state. Returns false, if
    /// no saved state exists on the stack.
    pub fn restore(&mut self) -> bool {
        if let Some(state) = self.states.pop() {
            self.set_state(state);
            true
        } else {
            false
        }
    }

    fn get_state(&self) -> State {
        State {
            from_node: self.current_from_node,
            to_node: self.current_to_node,
            edge: self.current_edge,
        }
    }

    fn set_state(&mut self, state: State) {
        self.current_from_node = state.from_node;
        self.current_to_node = state.to_node;
        self.current_edge = state.edge;
    }

    // XXX: Node function
    pub fn to_edge_list(&self) -> Vec<Vec<(usize, W)>> {
        self.nodes
            .iter()
            .map(|n| {
                n.out_edges
                 .iter()
                 .map(|oe| {
                     let edge = &self.edges[oe];
                     (edge.dst, edge.weight.clone())
                 })
                 .collect()
            })
            .collect()
    }

    pub fn apply_operation(&mut self, op: EdgeOperation<W, N>)
        where W: Neg<Output = W>
    {
        match op {
            EdgeOperation::IncreaseWeight  {weight: w} => self.update_edge_weight(w),
            EdgeOperation::DecreaseWeight  {weight: w} => self.update_edge_weight(-w),
            EdgeOperation::Duplicate       {weight: w} => self.duplicate(w),
            EdgeOperation::Split           {weight: w} => self.split(w),
            EdgeOperation::Loop            {weight: w} => self.add_self_loop(w),
            EdgeOperation::Output          {weight: w} => self.output(w),
            EdgeOperation::Merge           {n} => self.merge(n as usize),
            EdgeOperation::Next            {n} => self.next(n as usize),
            EdgeOperation::Parent          {n} => self.parent(n as usize),
            EdgeOperation::SetNodeFunction {function: f} => self.set_node_function(f),
            EdgeOperation::Reverse => self.reverse(),
            EdgeOperation::Save => self.save(),
            EdgeOperation::Restore => {
                let _ = self.restore();
            }
        }
    }

    /// creates an output node and connects it to the from neuron.
    /// does not change the link
    fn output(&mut self, weight: W) {
        let output_node = self.new_node();
        let from = self.current_from_node;
        let edge_idx = self.create_new_edge_with_weight(from, output_node, weight);
        self.insert_edge(edge_idx);
    }

    /// changes the node function of the current to-node.
    fn set_node_function(&mut self, node_fn: N) {
        self.nodes[self.current_to_node].node_fn = node_fn;
    }

    /// decrease-weight or increase-weight, depending on the sign of the weight.
    /// Updates the weight of the current edge, or in case of a virtual edge,
    /// creates a new edge with that weight.
    fn update_edge_weight(&mut self, weight: W) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let edge_idx = self.current_edge;

        // Update edge if it exists.
        if let Some(ref mut e) = self.edges.get_mut(&edge_idx) {
            assert!(e.src == from);
            assert!(e.dst == to);
            e.weight += weight;
            return;
        }

        // If it does not exist, create a new edge.

        let edge_idx = self.create_new_edge_with_weight(from, to, weight);
        self.insert_edge(edge_idx);
        self.current_edge = edge_idx;
    }

    /// Adds a loop to the current edge's target neuron.
    fn add_self_loop(&mut self, weight: W) {
        let to = self.current_to_node;
        let edge_idx = self.create_new_edge_with_weight(to, to, weight);
        self.insert_edge(edge_idx);
        self.current_edge = edge_idx;
        self.current_from_node = to;
    }

    /// Change from-node in current link to n-th sibling.
    /// This also delete the current edge, as after this operation,
    /// the edge between A->B is gone.
    fn next(&mut self, n: usize) {
        let new_from = (self.current_from_node + n) % self.nodes.len();
        self.change_from_node(new_from, true /* XXX */);
    }

    /// Change from-node in current link to n-th input edge of current from node.
    /// This also delete the current edge, as after this operation,
    /// the edge between A->B is gone.
    /// XXX: If current from node does not have any input edges? Delete node?
    /// XXX: Is the n-th input link removed?
    fn parent(&mut self, n: usize) {
        let new_from = {
            let from_node = &self.nodes[self.current_from_node];
            if from_node.in_edges.is_empty() {
                // ignore for now
                // XXX
                return;
            } else {
                let edge = from_node.in_edges[n % from_node.in_edges.len()];
                self.edges[&edge].src
            }
        };
        self.change_from_node(new_from, false);
    }

    /// Copy all incoming edges of from-node into to-node, then replace from-node with to-node.
    /// The current link is removed.
    fn merge(&mut self, n: usize) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let edge_idx = self.current_edge;
        let old_edge = self.edges.remove(&edge_idx);

        if let Some(e) = old_edge {
            // complete removal of edge
            assert!(e.src == from);
            assert!(e.dst == to);
            self.nodes[from].out_edges.retain(|&i| i != edge_idx);
            self.nodes[to].in_edges.retain(|&i| i != edge_idx);
        }

        if from != to {
            // copy all in-edges of from as in-edges into
            // as the edge target is changed, we create new edges.
            let old_in_edges = self.nodes[from].in_edges.clone();
            for eidx in old_in_edges {
                let old_edge = self.edges.remove(&eidx).unwrap();
                assert!(old_edge.dst == from);
                self.nodes[old_edge.src].out_edges.retain(|&i| i != eidx);
                self.nodes[old_edge.dst].in_edges.retain(|&i| i != eidx);
                let new_edge = if old_edge.src == old_edge.dst {
                    // we handle self-loops here.
                    self.create_new_edge_with_weight(to, to, old_edge.weight)
                } else {
                    self.create_new_edge_with_weight(old_edge.src, to, old_edge.weight)
                };

                self.insert_edge(new_edge);
            }

            // replace out-edges
            let old_out_edges = self.nodes[from].out_edges.clone();
            for eidx in old_out_edges {
                let old_edge = self.edges.remove(&eidx).unwrap();
                assert!(old_edge.src == from);
                if old_edge.src != old_edge.dst {
                    // we have handled self-loop already above
                    self.nodes[old_edge.src].out_edges.retain(|&i| i != eidx);
                    self.nodes[old_edge.dst].in_edges.retain(|&i| i != eidx);
                    let new_edge = self.create_new_edge_with_weight(to,
                                                                    old_edge.dst,
                                                                    old_edge.weight);
                    self.insert_edge(new_edge);
                }
            }
        }

        let edges = &self.nodes[to].in_edges;
        let l = edges.len();
        if l > 0 {
            // we use the n-th in edge as new current edge
            self.current_edge = edges[n % l];
            self.current_from_node = self.edges[&self.current_edge].src;
            assert!(self.edges[&self.current_edge].dst == to);
        } else {
            // we simply keep the current_edge index. it's invalid anyhow as the edge was
            // deleted.
        }
    }


    /// Split the current edge in two, and insert a node in the middle of it.
    /// If ```A -> B``` is the current edge, this will result into ```A -> N -> B```
    /// with ```N -> B``` being the next current edge.
    /// There is no A -> B link after this operation, that's why we delete the edge,
    /// so that backtracking cannot later make use of it.
    fn split(&mut self, weight: W) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let edge_idx = self.current_edge;

        // create intermediate node
        let middle_node = self.new_node();

        // remove the original edge (if it exists).
        let did_exist = self.edges.remove(&edge_idx);

        let orig_weight = did_exist.as_ref().map(|e| e.weight.clone());

        // then, insert two new edges.
        let first_edge = self.create_new_edge_with_weight(from,
                                                          middle_node,
                                                          orig_weight.unwrap_or(W::default())); // XXX: default?
        let second_edge = self.create_new_edge_with_weight(middle_node, to, weight);

        self.insert_or_update_edge(did_exist.map(|_| edge_idx),
                                   from,
                                   to,
                                   first_edge,
                                   second_edge);

        self.nodes[middle_node].in_edges.push(first_edge);
        self.nodes[middle_node].out_edges.push(second_edge);

        self.current_edge = second_edge;
        self.current_from_node = middle_node;
    }

    fn change_from_node(&mut self, new_from: usize, remove_existing_edge: bool) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let edge_idx = self.current_edge;

        let (did_exist, orig_weight) = if remove_existing_edge {
            // remove the original edge (if it exists).
            let did_exist = self.edges.remove(&edge_idx);

            let orig_weight = did_exist.as_ref().map(|e| e.weight.clone());

            (did_exist, orig_weight)
        } else {
            (None, None)
        };

        // add the new edge
        let new_edge = self.create_new_edge_with_weight(new_from,
                                                        to,
                                                        orig_weight.unwrap_or(W::default()));

        if let Some(_) = did_exist {
            // disconnect original from node
            self.nodes[from].out_edges.retain(|&e| e != edge_idx);
            // connect new from node
            self.nodes[new_from].out_edges.push(new_edge);

            // replace edge index in to node
            let mut ok = false;
            for e in self.nodes[to].in_edges.iter_mut() {
                if *e == edge_idx {
                    *e = new_edge;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
        } else {
            // for a virtual node, simply create it.
            self.insert_edge(new_edge);
        }

        self.current_edge = new_edge;
        self.current_from_node = new_from;
    }

    /// Duplicates the current edge
    fn duplicate(&mut self, weight: W) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let edge_id = self.create_new_edge_with_weight(from, to, weight);

        self.insert_or_update_edge(None, from, to, edge_id, edge_id);

        self.current_edge = edge_id;
    }

    /// Reverses the current edge
    fn reverse(&mut self) {
        let (from, to) = (self.current_from_node, self.current_to_node);
        let old_idx = self.current_edge;
        let current_edge = self.edges.remove(&old_idx);

        let weight = match current_edge {
            Some(ref c) => c.weight.clone(),
            None => W::default(), // virtual?
        };

        if let Some(e) = current_edge {
            assert!(e.src == from);
            assert!(e.dst == to);
            // remove
            self.nodes[from].out_edges.retain(|&e| e != old_idx);
            self.nodes[to].in_edges.retain(|&e| e != old_idx);
        }

        // insert new, reversed edge again.
        let new_edge_id = self.create_new_edge_with_weight(to, from, weight);

        self.insert_edge(new_edge_id);
        self.current_edge = new_edge_id;

        self.current_from_node = to;
        self.current_to_node = from;
    }

    fn insert_or_update_edge(&mut self,
                             old_edge: Option<usize>,
                             from: usize,
                             to: usize,
                             new_out_edge: usize,
                             new_in_edge: usize) {
        debug!("insert_or_update_edge(old_edge={:?},from={},to={},new_out_edge={},new_in_edge={})",
               old_edge,
               from,
               to,
               new_out_edge,
               new_in_edge);
        match old_edge {
            Some(old_id) => {
                self.replace_edge(old_id, from, to, new_out_edge, new_in_edge);
            }
            None => {
                self.nodes[from].out_edges.push(new_out_edge);
                self.nodes[to].in_edges.push(new_in_edge);
            }
        }
    }

    fn replace_edge(&mut self,
                    old_id: usize,
                    from: usize,
                    to: usize,
                    new_out_edge: usize,
                    new_in_edge: usize) {
        debug!("replace_edge(old_id={},from={},to={},new_out_edge={},new_in_edge={})",
               old_id,
               from,
               to,
               new_out_edge,
               new_in_edge);
        // replace `old_id` with the `new_out_edge` in the from node
        let mut ok;

        ok = false;
        for e in self.nodes[from].out_edges.iter_mut() {
            if *e == old_id {
                *e = new_out_edge;
                ok = true;
                break;
            }
        }
        assert!(ok);

        ok = false;
        // replace `old_id` with the `new_in_edge` in the to node
        for e in self.nodes[to].in_edges.iter_mut() {
            if *e == old_id {
                *e = new_in_edge;
                ok = true;
                break;
            }
        }
        assert!(ok);
    }

    fn insert_edge(&mut self, edge_idx: usize) {
        let edge = self.edges.get(&edge_idx).unwrap().clone();
        self.nodes[edge.src].out_edges.push(edge_idx);
        self.nodes[edge.dst].in_edges.push(edge_idx);
    }

    fn create_new_edge_with_weight(&mut self, from: usize, to: usize, weight: W) -> usize {
        let edge_id = self.next_edge_id;

        // self.nodes[from].out_edges.push(edge_id);
        // self.nodes[to].in_edges.push(edge_id);

        let mut edge = Edge::new(from, to);
        edge.weight = weight;
        self.next_edge_id += 1;
        self.edges.insert(edge_id, edge);
        return edge_id;
    }

    fn new_node(&mut self) -> usize {
        let node_id = self.nodes.len();
        let node = Node::new();
        self.nodes.push(node);
        return node_id;
    }
}

#[test]
fn test_empty() {
    let builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    let v: Vec<Vec<(usize, f32)>> = vec![vec![]];
    assert_eq!(v, builder.to_edge_list());
}

#[test]
fn test_split() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], builder.to_edge_list());

    builder.split(2.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(0, 2.0)]],
               builder.to_edge_list());

    builder.split(3.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               builder.to_edge_list());
}

#[test]
fn test_duplicate() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.duplicate(1.0);
    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], builder.to_edge_list());

    builder.split(0.25);
    assert_eq!(vec![vec![(0, 0.0), (1, 1.0)], vec![(0, 0.25)]],
               builder.to_edge_list());

    builder.duplicate(2.0);
    assert_eq!(vec![vec![(0, 0.0), (1, 1.0)], vec![(0, 0.25), (0, 2.0)]],
               builder.to_edge_list());
}

#[test]
fn test_reverse() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], builder.to_edge_list());

    builder.reverse();
    assert_eq!(vec![vec![(1, 0.0), (1, 1.0)], vec![]],
               builder.to_edge_list());
}

#[test]
fn test_update_edge_weight() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.update_edge_weight(4.0);
    assert_eq!(vec![vec![(0, 4.0)]], builder.to_edge_list());

    builder.update_edge_weight(-4.0);
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());
}

#[test]
fn test_add_self_loop() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.add_self_loop(1.0);
    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], builder.to_edge_list());
}

#[test]
fn test_next() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], builder.to_edge_list());

    builder.split(2.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(0, 2.0)]],
               builder.to_edge_list());

    builder.split(3.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               builder.to_edge_list());

    builder.next(1);
    assert_eq!(vec![vec![(1, 0.0), (0, 3.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![]],
               builder.to_edge_list());

    builder.next(3);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               builder.to_edge_list());
}

#[test]
fn test_parent() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.parent(0);
    assert_eq!(vec![vec![(0, 0.0), (0, 0.0)]], builder.to_edge_list());
    builder.parent(3);
    assert_eq!(vec![vec![(0, 0.0), (0, 0.0), (0, 0.0)]],
               builder.to_edge_list());

    builder.split(1.0);
    assert_eq!(vec![vec![(0, 0.0), (0, 0.0), (1, 0.0)], vec![(0, 1.0)]],
               builder.to_edge_list());

    builder.parent(0);
    assert_eq!(vec![vec![(0, 0.0), (0, 0.0), (1, 0.0), (0, 0.0)], vec![(0, 1.0)]],
               builder.to_edge_list());
}

#[test]
fn test_merge_self_loop() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.merge(1);
    let v: Vec<Vec<(usize, f32)>> = vec![vec![]];
    assert_eq!(v, builder.to_edge_list());
}

#[test]
fn test_merge_self_loop2() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.add_self_loop(1.0);

    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], builder.to_edge_list());

    builder.merge(1);
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());
}

#[test]
fn test_graph_paper() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();

    // figure 2.a
    builder.update_edge_weight(0.25);
    assert_eq!(vec![vec![(0, 0.25)]], builder.to_edge_list());

    // figure 2.b
    builder.split(0.8);
    assert_eq!(vec![vec![(1, 0.25)], vec![(0, 0.8)]],
               builder.to_edge_list());

    // figure 2.c
    builder.duplicate(3.0);
    assert_eq!(vec![vec![(1, 0.25)], vec![(0, 0.8), (0, 3.0)]],
               builder.to_edge_list());

    // figure 2.d
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (1, 3.0)], vec![(0, 0.8)]],
               builder.to_edge_list());

    // figure 2.e
    builder.split(0.8);
    builder.duplicate(2.0);
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)], vec![(0, 0.8), (2, 2.0)], vec![(1, 0.8)]],
               builder.to_edge_list());

    // figure 2.f
    builder.add_self_loop(1.0);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)], vec![(0, 0.8), (2, 2.0)], vec![(1, 0.8), (2, 1.0)]],
               builder.to_edge_list());

    // figure 2.g1
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6)]],
               builder.to_edge_list());

    // figure 2.g2
    builder.duplicate(0.4);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (2, 0.4)]],
               builder.to_edge_list());

    // figure 2.g3
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               builder.to_edge_list());

    // figure 2.g4
    builder.duplicate(0.4);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6), (2, 0.4)]],
               builder.to_edge_list());

    // figure 2.g5
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0), (4, 0.4)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               builder.to_edge_list());

    // figure 2.g
    builder.parent(1);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0), (4, 0.0)],
                    vec![(1, 0.8), (3, 1.0), (4, 0.4)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               builder.to_edge_list());

    // figure 2.h
    builder.merge(1);
    assert_eq!(vec![vec![(2, 3.0), (4, 0.25)],
                    vec![],
                    vec![(3, 1.0), (4, 0.4), (4, 0.8)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6), (0, 0.8), (2, 2.0)]],
               builder.to_edge_list());

}

#[test]
fn test_save_restore() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.save();
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.split(0.5);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], builder.to_edge_list());

    assert_eq!(true, builder.restore());
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], builder.to_edge_list());

    // there is now a virtual edge between 0 -> 0.
    builder.update_edge_weight(0.7); // this will create a real edge.

    assert_eq!(vec![vec![(1, 0.0), (0, 0.7)], vec![(0, 0.5)]],
               builder.to_edge_list());

    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.0), (2, 0.7)], vec![(0, 0.5)], vec![(0, 0.6)]],
               builder.to_edge_list());
}

#[test]
fn test_save_restore2() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.save();
    assert_eq!(vec![vec![(0, 0.0)]], builder.to_edge_list());

    builder.split(0.5);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], builder.to_edge_list());

    assert_eq!(true, builder.restore());
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], builder.to_edge_list());

    // there is now a virtual edge between 0 -> 0 with weight 0.0.
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.0), (2, 0.0)], vec![(0, 0.5)], vec![(0, 0.6)]],
               builder.to_edge_list());
}
