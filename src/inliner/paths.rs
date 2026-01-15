use std::path::{Path, PathBuf};

pub fn resolve_subgraph_path(current_file: &Path, target: &str) -> PathBuf {
    // 1. Если путь начинается с assets/, он абсолютный от корня проекта
    if target.starts_with("assets/") {
        let mut p = PathBuf::from(target);
        if !p.to_string_lossy().ends_with(".json") {
            p.set_extension("json");
        }
        return p;
    }

    // 2. Иначе пробуем относительно текущего файла
    let mut p = current_file.parent().unwrap_or_else(|| Path::new(".")).join(target);
    if !p.to_string_lossy().ends_with(".json") {
        p.set_extension("json");
    }

    // 3. Если относительно файла не нашли, пробуем в библиотеке (assets/lib)
    if !p.exists() {
        let mut lib_p = PathBuf::from("assets/lib").join(target);
        if !lib_p.to_string_lossy().ends_with(".json") {
            lib_p.set_extension("json");
        }
        if lib_p.exists() {
            return lib_p;
        }
    }

    p
}