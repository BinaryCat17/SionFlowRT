use crate::model::{Dimension, TensorShape};

pub struct ShapeEngine;

impl ShapeEngine {
    /// Упрощает символьное выражение (например, N + 0 -> N)
    pub fn simplify(dim: Dimension) -> Dimension {
        match dim {
            Dimension::Add(l, r) => {
                let l = Self::simplify(*l);
                let r = Self::simplify(*r);
                match (&l, &r) {
                    (Dimension::Value(0), _) => r,
                    (_, Dimension::Value(0)) => l,
                    (Dimension::Value(a), Dimension::Value(b)) => Dimension::Value(a + b),
                    _ => Dimension::Add(Box::new(l), Box::new(r)),
                }
            }
            Dimension::Mul(l, r) => {
                let l = Self::simplify(*l);
                let r = Self::simplify(*r);
                match (&l, &r) {
                    (Dimension::Value(1), _) => r,
                    (_, Dimension::Value(1)) => l,
                    (Dimension::Value(0), _) | (_, Dimension::Value(0)) => Dimension::Value(0),
                    (Dimension::Value(a), Dimension::Value(b)) => Dimension::Value(a * b),
                    _ => Dimension::Mul(Box::new(l), Box::new(r)),
                }
            }
            _ => dim,
        }
    }

    /// Унифицирует две формы (Broadcasting). 
    /// Возвращает результирующую форму или ошибку несовместимости.
    pub fn unify(s1: &TensorShape, s2: &TensorShape) -> anyhow::Result<TensorShape> {
        let d1 = Self::expand_ellipsis_from_target(&s1.dims, &s2.dims);
        let d2 = Self::expand_ellipsis_from_target(&s2.dims, &d1);

        let mut res = Vec::new();
        let (mut i, mut j) = (d1.len() as isize - 1, d2.len() as isize - 1);

        while i >= 0 || j >= 0 {
            let v1 = if i >= 0 { Some(&d1[i as usize]) } else { None };
            let v2 = if j >= 0 { Some(&d2[j as usize]) } else { None };

            match (v1, v2) {
                (Some(a), Some(b)) => {
                    res.push(Self::unify_dims(a, b)?);
                }
                (Some(a), None) | (None, Some(a)) => {
                    res.push(a.clone());
                }
                _ => {}
            }
            i -= 1;
            j -= 1;
        }

        res.reverse();
        Ok(TensorShape { dims: res })
    }

    fn unify_dims(d1: &Dimension, d2: &Dimension) -> anyhow::Result<Dimension> {
        if d1 == d2 { return Ok(d1.clone()); }
        if d1.is_wildcard() { return Ok(d2.clone()); }
        if d2.is_wildcard() { return Ok(d1.clone()); }
        
        match (d1, d2) {
            (Dimension::Value(1), other) | (other, Dimension::Value(1)) => Ok(other.clone()),
            (Dimension::Value(a), Dimension::Value(b)) if a != b => {
                Err(anyhow::anyhow!("Incompatible fixed dimensions: {} and {}", a, b))
            }
            _ => {
                Ok(d1.clone()) 
            }
        }
    }

    /// Раскрывает Ellipsis (...) используя информацию о ранге другой формы
    pub fn expand_ellipsis_from_target(dims: &[Dimension], target: &[Dimension]) -> Vec<Dimension> {
        let ellipsis_pos = dims.iter().position(|d| d.is_ellipsis());
        
        if let Some(pos) = ellipsis_pos {
            let mut result = Vec::new();
            for i in 0..pos { result.push(dims[i].clone()); }
            
            let after_count = dims.len() - 1 - pos;
            let needed = target.len().saturating_sub(after_count).saturating_sub(pos);
            
            for i in 0..needed {
                if let Some(target_dim) = target.get(pos + i) {
                    result.push(target_dim.clone());
                } else {
                    result.push(Dimension::Value(1));
                }
            }
            
            for i in (pos + 1)..dims.len() { result.push(dims[i].clone()); }
            result
        } else {
            dims.to_vec()
        }
    }

    /// Вычисляет объем формы (произведение всех размерностей)
    pub fn volume(shape: &TensorShape) -> Dimension {
        let mut res = Dimension::Value(1);
        for d in &shape.dims {
            res = Dimension::Mul(Box::new(res), Box::new(d.clone()));
        }
        Self::simplify(res)
    }
}