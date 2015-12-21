#![feature(augmented_assignments)]
#![feature(op_assign_traits)]

use std::fmt::Debug;
use std::collections::BTreeMap;
use std::ops::AddAssign;
use std::ops::Neg;

pub trait NthEdge: Clone {
    fn edge_index(&self, num_edges: usize, offset: usize) -> Option<usize>;
}

#[derive(Debug, Clone, Copy)]
pub struct NthEdgeI(pub u32);
#[derive(Debug, Clone, Copy)]
pub struct NthEdgeF(pub f32);

impl NthEdge for NthEdgeI {
    fn edge_index(&self, num_edges: usize, offset: usize) -> Option<usize> {
        if num_edges > 0 {
            Some((self.0 as usize + offset) % num_edges)
        } else {
            None
        }
    }
}

impl NthEdge for NthEdgeF {
    fn edge_index(&self, num_edges: usize, offset: usize) -> Option<usize> {
        debug_assert!(self.0 >= 0.0 && self.0 < 1.0);
        if num_edges > 0 {
            let edge = (self.0 * num_edges as f32) as usize;
            debug_assert!(edge < num_edges);
            Some((edge + offset) % num_edges)
        } else {
            None
        }
    }
}

#[test]
fn test_nthedge() {
    let n = NthEdgeI(3);
    assert_eq!(Some(3), n.edge_index(4, 0));
    assert_eq!(Some(0), n.edge_index(3, 0));
    assert_eq!(None, n.edge_index(0, 0));

    assert_eq!(Some(0), NthEdgeF(0.0).edge_index(10, 0));
    assert_eq!(Some(0), NthEdgeF(0.09).edge_index(10, 0));
    assert_eq!(Some(1), NthEdgeF(0.1).edge_index(10, 0));
    assert_eq!(Some(2), NthEdgeF(0.2).edge_index(10, 0));
    assert_eq!(Some(3), NthEdgeF(0.3).edge_index(10, 0));
    assert_eq!(Some(9), NthEdgeF(0.9).edge_index(10, 0));
    assert_eq!(Some(9), NthEdgeF(0.999).edge_index(10, 0));
    assert_eq!(Some(9), NthEdgeF(0.999).edge_index(10, 10));
}

#[derive(Debug, Clone)]
pub enum EdgeOperation<W: Clone, N: Clone, NT: Clone> {
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
        n: NT,
    },
    Next {
        n: NT,
    },
    Parent {
        n: NT,
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

impl<W: Debug + Clone + Default> Edge<W> {
    fn new(src: usize, dst: usize, weight: W) -> Edge<W> {
        Edge {
            src: src,
            dst: dst,
            weight: weight,
        }
    }
}

#[derive(Debug, Clone)]
struct Node<N: Clone + Default + Debug> {
    in_edges: Vec<usize>,
    out_edges: Vec<usize>,
    node_fn: N,
}

impl<N: Default + Clone + Debug> Node<N> {
    fn new() -> Node<N> {
        Node {
            in_edges: Vec::new(),
            out_edges: Vec::new(),
            node_fn: N::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.in_edges.is_empty() && self.out_edges.is_empty()
    }
}

#[derive(Debug, Clone)]
struct State {
    from_node: usize,
    to_node: usize,
    link_in_to_node: usize,
}

#[derive(Debug)]
pub struct GraphBuilder<W: Debug + Default + Clone, N: Debug + Default + Clone> {
    edges: BTreeMap<usize, Edge<W>>,
    nodes: Vec<Node<N>>,
    next_edge_id: usize,
    current_state: State,
    states: Vec<State>,
}

impl<W: Debug + Default + Clone + AddAssign<W>, N: Debug + Default + Clone> GraphBuilder<W, N> {
    pub fn new() -> GraphBuilder<W, N> {
        // we begin with an empty node and a virtual edge.

        GraphBuilder {
            edges: BTreeMap::new(),
            nodes: vec![Node::new()],
            next_edge_id: 0,

            // link_in_to_node: 0 points to a non-existing "virtual" edge
            current_state: State {
                from_node: 0,
                to_node: 0,
                link_in_to_node: 0,
            },
            states: Vec::new(),
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
        self.current_state.clone()
    }

    fn set_state(&mut self, state: State) {
        self.current_state = state;
    }

    pub fn count_nodes(&self) -> usize {
        let mut count = 0;
        self.visit_nodes(|_, _| count += 1);
        count
    }

    // iterate over all non-empty nodes.
    #[inline]
    pub fn visit_nodes<F: FnMut(usize, &N)>(&self, mut callback: F) {
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.is_empty() {
                callback(i, &node.node_fn);
            }
        }
    }

    #[inline]
    pub fn visit_edges<F: FnMut((usize, usize), &W)>(&self, mut callback: F) {
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.is_empty() {
                for out_edge in node.out_edges.iter() {
                    let edge = &self.edges[out_edge];
                    debug_assert!(edge.src == i);
                    callback((edge.src, edge.dst), &edge.weight);
                }
            }
        }
    }

    pub fn to_edge_list(&self) -> Vec<Option<(N, Vec<(usize, W)>)>> {
        self.nodes
            .iter()
            .map(|n| {
                if n.is_empty() {
                    None
                } else {
                    Some((n.node_fn.clone(),
                          n.out_edges
                           .iter()
                           .map(|oe| {
                               let edge = &self.edges[oe];
                               (edge.dst, edge.weight.clone())
                           })
                           .collect()))
                }
            })
            .collect()
    }

    pub fn apply_operation<NT: NthEdge>(&mut self, op: EdgeOperation<W, N, NT>)
        where W: Neg<Output = W>
    {
        match op {
            EdgeOperation::IncreaseWeight  {weight: w} => self.update_edge_weight(w),
            EdgeOperation::DecreaseWeight  {weight: w} => self.update_edge_weight(-w),
            EdgeOperation::Duplicate       {weight: w} => self.duplicate(w),
            EdgeOperation::Split           {weight: w} => self.split(w),
            EdgeOperation::Loop            {weight: w} => self.add_self_loop(w),
            EdgeOperation::Output          {weight: w} => self.output(w),
            EdgeOperation::Merge           {n} => self.merge(n),
            EdgeOperation::Next            {n} => self.next(n),
            EdgeOperation::Parent          {n} => self.parent(n),
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
        let from = self.current_state.from_node;
        let output_node = self.new_node();
        let edge_idx = self.create_new_edge_with_weight(from, output_node, weight);
        self.insert_edge(edge_idx);
    }

    /// changes the node function of the current to-node.
    fn set_node_function(&mut self, node_fn: N) {
        self.nodes[self.current_state.to_node].node_fn = node_fn;
    }

    fn get_current_edge(&self) -> Option<usize> {
        let to_node = &self.nodes[self.current_state.to_node];
        if let Some(&edge_idx) = to_node.in_edges.get(self.current_state.link_in_to_node) {
            let edge = &self.edges[&edge_idx];
            if edge.src == self.current_state.from_node && edge.dst == self.current_state.to_node {
                Some(edge_idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// decrease-weight or increase-weight, depending on the sign of the weight.
    /// Updates the weight of the current edge, or in case of a virtual edge,
    /// creates a new edge with that weight.
    pub fn update_edge_weight(&mut self, weight: W) {
        let (from, to) = (self.current_state.from_node, self.current_state.to_node);
        let cur_edge = self.get_current_edge();

        if let Some(edge_idx) = cur_edge {
            let e = self.get_mut_edge(edge_idx);
            assert!(e.src == from);
            assert!(e.dst == to);
            e.weight += weight;
        } else {
            // Virtual edge. Create!
            let edge_idx = self.create_new_edge_with_weight(from, to, weight);
            self.current_state.link_in_to_node = self.nodes[to].in_edges.len();
            self.insert_edge(edge_idx);
        }
    }

    /// Adds a loop to the current edge's target neuron.
    pub fn add_self_loop(&mut self, weight: W) {
        let to = self.current_state.to_node;
        let edge_idx = self.create_new_edge_with_weight(to, to, weight);
        self.current_state.link_in_to_node = self.nodes[to].in_edges.len();
        self.current_state.from_node = to;
        self.insert_edge(edge_idx);
    }

    fn get_mut_edge(&mut self, edge_idx: usize) -> &mut Edge<W> {
        self.edges.get_mut(&edge_idx).unwrap()
    }

    /// Change from-node of current link to it's n-th sibling.
    /// The n-th sibling is the current+n-th incoming node into the to-node.
    pub fn next<NT: NthEdge>(&mut self, n: NT) {
        let to_node = &self.nodes[self.current_state.to_node];
        if let Some(new_n) = n.edge_index(to_node.in_edges.len(),
                                          self.current_state.link_in_to_node) {
            let sibling_edge = to_node.in_edges[new_n];
            let new_from = self.edges[&sibling_edge].src;
            self.current_state.from_node = new_from;
            self.current_state.link_in_to_node = new_n;
        }
    }

    /// Change from-node of current link to n-th input edge of current from node.
    /// If no input edge exists, the edge is left as is. TODO: Test case
    /// The n-th input node is not deleted.
    /// NOTE: Does not modify the graph itself, only changes the current link.
    pub fn parent<NT: NthEdge>(&mut self, n: NT) {
        let from = self.current_state.from_node;
        let edge_idx = n.edge_index(self.nodes[from].in_edges.len(), 0);
        if let Some(idx) = edge_idx {
            let new_from = self.edges[&self.nodes[from].in_edges[idx]].src;
            self.current_state.from_node = new_from;
            self.current_state.link_in_to_node = idx;
        } else {
            // XXX
            // self.current_state.link_in_to_node = 0;
            return;
        }
    }

    /// Copy all incoming edges of from-node into to-node, then replace from-node with to-node.
    /// The current link is removed.
    /// Do not create a self-loop for links between A and B.
    fn merge<NT: NthEdge>(&mut self, n: NT) {
        // 1. delete current edge.
        // 2. modify all outgoing edges of A. replace the .src to B.
        // 3. copy all incoming edges of A to B.
        // 4. set the n-th incoming edge into A as new current edge.

        let (from, to) = (self.current_state.from_node, self.current_state.to_node);

        // 1. delete current edge (if exists)
        if let Some(edge_idx) = self.get_current_edge() {
            let edge = self.edges.remove(&edge_idx).unwrap();
            debug_assert!(edge.src == from);
            debug_assert!(edge.dst == to);
            self.nodes[from].out_edges.retain(|&i| i != edge_idx);
            self.nodes[to].in_edges.retain(|&i| i != edge_idx);
        }

        if from == to {
            return;
        }

        // 2. modify all outgoing edges of A.
        let mut new_out_edges = Vec::new();
        for &out_edge in self.nodes[from].out_edges.iter() {
            let edge = self.edges.get_mut(&out_edge).unwrap();
            debug_assert!(edge.src == from);
            edge.src = to;
            new_out_edges.push(out_edge);
        }
        self.nodes[to].out_edges.extend(new_out_edges);

        // 3. copy all incoming edges of A to B. Modify them.
        let mut new_in_edges = Vec::new();
        for &in_edge in self.nodes[from].in_edges.iter() {
            let edge = self.edges.get_mut(&in_edge).unwrap();
            debug_assert!(edge.dst == from);
            edge.dst = to;
            new_in_edges.push(in_edge);
        }
        self.nodes[to].in_edges.extend(new_in_edges);

        // remove from node (TODO: improve)
        self.nodes[from].out_edges.clear();
        self.nodes[from].in_edges.clear();

        // 4.
        let edges = &self.nodes[to].in_edges;
        if let Some(idx) = n.edge_index(edges.len(), 0) {
            // we use the n-th in edge as new current edge
            self.current_state.link_in_to_node = edges[idx];
            self.current_state.from_node = self.edges[&self.current_state.link_in_to_node].src;
            debug_assert!(self.edges[&self.current_state.link_in_to_node].dst == to);
        } else {
            self.current_state.link_in_to_node = 0;
            self.current_state.from_node = to;
        }
    }

    /// Split the current edge in two, and insert a node in the middle of it.
    /// If ```A -> B``` is the current edge, this will result into ```A -> N -> B```
    /// with ```N -> B``` being the next current edge.
    /// There is no A -> B link after this operation, that's why we delete the edge,
    /// so that backtracking cannot later make use of it.
    fn split(&mut self, weight: W) {
        let (from, to) = (self.current_state.from_node, self.current_state.to_node);

        // create intermediate node
        let middle_node = self.new_node();

        // Move current link from (A,B) to (A,C) (or create a new with weight 0.0).
        let cur_edge = self.get_current_edge();
        if let Some(edge_idx) = cur_edge {
            // new to_node is middle_node.
            self.get_mut_edge(edge_idx).dst = middle_node;

            // remove from incoming edges.
            self.nodes[to].in_edges.retain(|&i| i != edge_idx);
            // add to incoming edges of new to_node
            self.nodes[middle_node].in_edges.push(edge_idx);
        } else {
            // current edge is virtual. Create a edge with weight 0.0.
            let edge_idx = self.create_new_edge_with_weight(from, middle_node, W::default());
            self.insert_edge(edge_idx);
        }

        // Add new link from middle_node to `B` with `weight`
        let edge_idx = self.create_new_edge_with_weight(middle_node, to, weight);
        self.current_state.link_in_to_node = self.insert_edge(edge_idx);
        self.current_state.from_node = middle_node;
    }

    /// Duplicates the current edge. The new edge becomes the new current edge.
    fn duplicate(&mut self, weight: W) {
        let (from, to) = (self.current_state.from_node, self.current_state.to_node);
        let edge_idx = self.create_new_edge_with_weight(from, to, weight);
        self.current_state.link_in_to_node = self.insert_edge(edge_idx);
    }

    /// Reverse the current edge.
    fn reverse(&mut self) {
        let (from, to) = (self.current_state.from_node, self.current_state.to_node);

        // Delete current link
        let cur_edge = self.get_current_edge();
        let orig_weight = if let Some(edge_idx) = cur_edge {
            // delete edge
            let edge = self.edges.remove(&edge_idx).unwrap();
            self.nodes[from].out_edges.retain(|&i| i != edge_idx);
            self.nodes[to].in_edges.retain(|&i| i != edge_idx);
            edge.weight
        } else {
            // If current edge does not exist, just change the state.
            W::default()
        };

        // Add new reversed link
        let edge_idx = self.create_new_edge_with_weight(to, from, orig_weight);
        let new_state = State {
            link_in_to_node: self.insert_edge(edge_idx),
            to_node: from,
            from_node: to,
        };
        self.current_state = new_state;
    }

    /// Returns the incoming edge index of the target node.
    fn insert_edge(&mut self, edge_idx: usize) -> usize {
        let edge = self.edges.get(&edge_idx).unwrap().clone(); // XXX
        self.nodes[edge.src].out_edges.push(edge_idx);
        let idx = self.nodes[edge.dst].in_edges.len();
        self.nodes[edge.dst].in_edges.push(edge_idx);
        idx
    }

    /// Allocates a new Edge and returns new edge index.
    fn create_new_edge_with_weight(&mut self, from: usize, to: usize, weight: W) -> usize {
        let edge_id = self.next_edge_id;
        let edge = Edge::new(from, to, weight);
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

#[cfg(test)]
fn edge_list<W: Debug + Default + Clone + AddAssign<W>, N: Debug + Default + Clone>
    (builder: &GraphBuilder<W, N>)
     -> Vec<Vec<(usize, W)>> {
    builder.to_edge_list()
           .into_iter()
           .map(|n| {
               match n {
                   Some((_, b)) => b,
                   None => vec![],
               }
           })
           .collect()
}

#[test]
fn test_empty() {
    let builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    let v: Vec<Vec<(usize, f32)>> = vec![vec![]];
    assert_eq!(v, edge_list(&builder));
}

#[test]
fn test_split() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], edge_list(&builder));

    builder.split(2.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(0, 2.0)]],
               edge_list(&builder));

    builder.split(3.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               edge_list(&builder));
}

#[test]
fn test_duplicate() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.duplicate(1.0);
    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], edge_list(&builder));

    builder.split(0.25);
    assert_eq!(vec![vec![(0, 0.0), (1, 1.0)], vec![(0, 0.25)]],
               edge_list(&builder));

    builder.duplicate(2.0);
    assert_eq!(vec![vec![(0, 0.0), (1, 1.0)], vec![(0, 0.25), (0, 2.0)]],
               edge_list(&builder));
}

#[test]
fn test_reverse() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], edge_list(&builder));

    builder.reverse();
    assert_eq!(vec![vec![(1, 0.0), (1, 1.0)], vec![]], edge_list(&builder));
}

#[test]
fn test_update_edge_weight() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.update_edge_weight(4.0);
    assert_eq!(vec![vec![(0, 4.0)]], edge_list(&builder));

    builder.update_edge_weight(-4.0);
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));
}

#[test]
fn test_add_self_loop() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.add_self_loop(1.0);
    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], edge_list(&builder));
}

#[test]
fn test_next() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], edge_list(&builder));

    builder.split(2.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(0, 2.0)]],
               edge_list(&builder));

    builder.split(3.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               edge_list(&builder));

    // does not change anything, because there is only one edge (no sibling).
    builder.next(NthEdgeI(1));
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0)]],
               edge_list(&builder));

    builder.duplicate(5.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0), (0, 5.0)]],
               edge_list(&builder));

    builder.update_edge_weight(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0), (0, 6.0)]],
               edge_list(&builder));

    builder.next(NthEdgeI(0));
    builder.update_edge_weight(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 3.0), (0, 7.0)]],
               edge_list(&builder));


    builder.next(NthEdgeI(1));
    builder.update_edge_weight(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 4.0), (0, 7.0)]],
               edge_list(&builder));

    builder.next(NthEdgeI(2));
    builder.update_edge_weight(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(2, 1.0)], vec![(3, 2.0)], vec![(0, 5.0), (0, 7.0)]],
               edge_list(&builder));
}

#[test]
fn test_parent() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.parent(NthEdgeI(0));
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));
    builder.parent(NthEdgeI(3));
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(1.0);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], edge_list(&builder));

    builder.parent(NthEdgeI(0));
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 1.0)]], edge_list(&builder));

    // XXX: add more tests
}

#[test]
fn test_merge_self_loop() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.merge(NthEdgeI(1));
    let v: Vec<Vec<(usize, f32)>> = vec![vec![]];
    assert_eq!(v, edge_list(&builder));

    let mut nodes = vec![];
    builder.visit_nodes(|i, _| nodes.push(i));
    assert!(nodes.is_empty());
    assert_eq!(0, builder.count_nodes());

    let mut edges = vec![];
    builder.visit_edges(|(i, j), _w| edges.push((i, j)));
    let res: Vec<(usize, usize)> = vec![];
    assert_eq!(res, edges);
}

#[test]
fn test_merge_self_loop2() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.add_self_loop(1.0);

    assert_eq!(vec![vec![(0, 0.0), (0, 1.0)]], edge_list(&builder));

    builder.merge(NthEdgeI(1));
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));
}

#[test]
fn test_graph_paper() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();

    // figure 2.a
    builder.update_edge_weight(0.25);
    assert_eq!(vec![vec![(0, 0.25)]], edge_list(&builder));

    // figure 2.b
    builder.split(0.8);
    assert_eq!(vec![vec![(1, 0.25)], vec![(0, 0.8)]], edge_list(&builder));

    // figure 2.c
    builder.duplicate(3.0);
    assert_eq!(vec![vec![(1, 0.25)], vec![(0, 0.8), (0, 3.0)]],
               edge_list(&builder));

    // figure 2.d
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (1, 3.0)], vec![(0, 0.8)]],
               edge_list(&builder));

    // figure 2.e
    builder.split(0.8);
    builder.duplicate(2.0);
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)], vec![(0, 0.8), (2, 2.0)], vec![(1, 0.8)]],
               edge_list(&builder));

    // figure 2.f
    builder.add_self_loop(1.0);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)], vec![(0, 0.8), (2, 2.0)], vec![(1, 0.8), (2, 1.0)]],
               edge_list(&builder));

    // figure 2.g1
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6)]],
               edge_list(&builder));

    // figure 2.g2
    builder.duplicate(0.4);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (2, 0.4)]],
               edge_list(&builder));

    // figure 2.g3
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               edge_list(&builder));

    // figure 2.g4
    builder.duplicate(0.4);
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6), (2, 0.4)]],
               edge_list(&builder));

    // figure 2.g5
    builder.reverse();
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0), (4, 0.4)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               edge_list(&builder));

    assert_eq!(4, builder.get_state().to_node);
    assert_eq!(2, builder.get_state().from_node);
    assert_eq!(1, builder.get_state().link_in_to_node);

    // figure 2.g
    builder.parent(NthEdgeI(1));
    assert_eq!(vec![vec![(1, 0.25), (2, 3.0)],
                    vec![(0, 0.8), (2, 2.0)],
                    vec![(1, 0.8), (3, 1.0), (4, 0.4)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6)]],
               edge_list(&builder));

    assert_eq!(4, builder.get_state().to_node);
    assert_eq!(1, builder.get_state().from_node);
    assert_eq!(1, builder.get_state().link_in_to_node);

    // figure 2.h
    builder.merge(NthEdgeI(1));
    assert_eq!(vec![vec![(4, 0.25), (2, 3.0)],
                    vec![],
                    vec![(4, 0.8), (3, 1.0), (4, 0.4)],
                    vec![(2, 0.6), (4, 0.4)],
                    vec![(2, 0.6), (0, 0.8), (2, 2.0)]],
               edge_list(&builder));

}

#[test]
fn test_save_restore() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.save();
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(0.5);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], edge_list(&builder));

    assert_eq!(true, builder.restore());
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], edge_list(&builder));

    // there is now a virtual edge between 0 -> 0.
    builder.update_edge_weight(0.7); // this will create a real edge.

    assert_eq!(vec![vec![(1, 0.0), (0, 0.7)], vec![(0, 0.5)]],
               edge_list(&builder));

    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.0), (2, 0.7)], vec![(0, 0.5)], vec![(0, 0.6)]],
               edge_list(&builder));
}

#[test]
fn test_save_restore2() {
    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    builder.update_edge_weight(0.0);

    // start with a single node, self-connected with zero weight
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.save();
    assert_eq!(vec![vec![(0, 0.0)]], edge_list(&builder));

    builder.split(0.5);
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], edge_list(&builder));

    assert_eq!(true, builder.restore());
    assert_eq!(vec![vec![(1, 0.0)], vec![(0, 0.5)]], edge_list(&builder));

    // there is now a virtual edge between 0 -> 0 with weight 0.0.
    builder.split(0.6);
    assert_eq!(vec![vec![(1, 0.0), (2, 0.0)], vec![(0, 0.5)], vec![(0, 0.6)]],
               edge_list(&builder));
}
