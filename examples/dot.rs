extern crate grabinput;
extern crate asexp;
extern crate graph_edge_evolution; 

use asexp::Sexp;
use graph_edge_evolution::{NthEdgeF, EdgeOperation, GraphBuilder, EdgeType};
use std::fs::File;
use std::io::Write;

fn main() {
    let s = grabinput::all(std::env::args().nth(1)); 
    let expr = Sexp::parse(&s).unwrap();
    let ops: Vec<EdgeOperation<f32, usize, NthEdgeF>> = expr.get_vec(|op| {
        match op {
            &Sexp::Tuple(ref elms) => {
                let arg = elms[1].get_vec(|f| f.get_float()).unwrap()[0] as f32;
                Some(match elms[0].get_str().unwrap() {
                    "Duplicate" => EdgeOperation::Duplicate {weight: arg},
                    "Split" => EdgeOperation::Split {weight: arg},
                    "Loop" => EdgeOperation::Loop {weight: arg},
                    "Output" => EdgeOperation::Output {weight: arg},

                    "Merge" => EdgeOperation::Merge {n: NthEdgeF(arg)},
                    "Next" => EdgeOperation::Next {n: NthEdgeF(arg)},
                    "Parent" => EdgeOperation::Parent {n: NthEdgeF(arg)},
                    "Reverse" => EdgeOperation::Reverse,
                    "Save" => EdgeOperation::Save,
                    "Restore" => EdgeOperation::Save,
                    _ => return None,
                })
            }
            _ => return None
        }
    }).unwrap();
    println!("{:?}", expr);
    println!("{:?}", ops);

    let mut builder: GraphBuilder<f32, usize> = GraphBuilder::new();
    for (i, op) in ops.iter().enumerate() {
        let op = op.clone();
        println!("{}: {:?}", i, op);
        builder.apply_operation(op);

        let mut f = File::create(format!("graph_{:02}.dot", i)).unwrap();
        writeln!(&mut f, "digraph Graph{:02} {{ node[shape=circle]; rankdir=TB; splines=curved;", i); 

        builder.visit_nodes(|i, _| writeln!(&mut f, "{};", i).unwrap());
        builder.visit_edges_with_type(|(i, j), _w, edge_type| {
            // penwidth=2
            let style = match edge_type {
                EdgeType::Active => "[color=red,penwidth=2]",
                EdgeType::Virtual => "[color=red,style=dashed,penwidth=2]",  
                EdgeType::Normal => "[color=black]",
            };

            writeln!(&mut f, "{} -> {} {};", i, j, style).unwrap()
        });

        writeln!(&mut f, "}}");
    }
}
