pub fn sanitize_id(id: &str) -> String {
    id.replace("/", "_").replace(".", "_")
}