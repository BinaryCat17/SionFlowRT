use crate::pipeline::{Stage, CompilerContext};
use std::fs;
use std::process::Command;

pub struct BuildStage;
impl Stage for BuildStage {
    fn name(&self) -> &str { "Emit & GCC Build" }
    fn run(&self, ctx: &mut CompilerContext) -> anyhow::Result<()> {
        fs::create_dir_all(&ctx.gen_dir)?;
        fs::create_dir_all(&ctx.out_dir)?;

        let module = ctx.generated_module.as_ref().ok_or_else(|| anyhow::anyhow!("No module code generated"))?;
        let runtime = ctx.generated_runtime.as_ref().ok_or_else(|| anyhow::anyhow!("No runtime code generated"))?;
        
        fs::write(ctx.gen_dir.join("generated.c"), module)?;
        fs::write(ctx.gen_dir.join("runtime.c"), runtime)?;

        println!("Compiling C code...");
        let bin_path = ctx.out_dir.join("generated_bin");
        let status = Command::new("gcc")
            .args(["-O3", "-fopenmp", 
                   &ctx.gen_dir.join("generated.c").to_string_lossy(), 
                   &ctx.gen_dir.join("runtime.c").to_string_lossy(), 
                   "-o", &bin_path.to_string_lossy(), 
                   "-lSDL2", "-lm"])
            .status()?;

        if !status.success() {
            return Err(anyhow::anyhow!("C compilation failed"));
        }
        println!("Build successful: {}", bin_path.display());
        Ok(())
    }
}
