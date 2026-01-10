mod model;
mod compiler;
mod codegen_c;

use model::ComputationalGraph;
use compiler::Compiler;
use codegen_c::CodegenC;
use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let asset_path = "assets/example_graph.json";
    let gen_dir = "generated";
    let out_dir = "out";
    let log_dir = "logs";

    for dir in &[gen_dir, out_dir, log_dir] {
        fs::create_dir_all(dir)?;
    }

    println!("1. Загрузка графа из {}...", asset_path);
    let json_content = fs::read_to_string(asset_path)?;
    
    let raw_graph = ComputationalGraph::from_json(&json_content)?;
    let mut compiler = Compiler::new();
    let execution_order = compiler.build(raw_graph)?;
    
    println!("2. Граф построен. Генерация C-кода через Tera...");
    
    let codegen = CodegenC::new(&compiler);
    let c_code = codegen.generate(&execution_order)?;
    
    let c_file_path = Path::new(gen_dir).join("generated.c");
    let mut file = File::create(&c_file_path)?;
    file.write_all(c_code.as_bytes())?;
    println!("3. Код C сгенерирован.");

    let bin_path = Path::new(out_dir).join("generated_bin");
    let log_path = Path::new(log_dir).join("build.log");

    println!("4. Компиляция через gcc...");
    let output = Command::new("gcc")
        .arg("-O3")
        .arg("-fopenmp")
        .arg(&c_file_path)
        .arg("-o")
        .arg(&bin_path)
        .arg("-lm")
        .output()?;

    fs::write(log_path, format!("STDOUT:\n{}\nSTDERR:\n{}", 
        String::from_utf8_lossy(&output.stdout), 
        String::from_utf8_lossy(&output.stderr)))?;

    if output.status.success() {
        println!("5. Сборка успешна. Запуск...");
        let run_out = Command::new(&bin_path).output()?;
        println!("--- Вывод программы ---");
        println!("{}", String::from_utf8_lossy(&run_out.stdout));
    } else {
        println!("ОШИБКА: Компиляция не удалась. Смотрите logs/build.log");
    }

    Ok(())
}
