"""
Sistema de Recomendación basado en Perfiles de Usuario
Combina historial implícito, feedback explícito y preguntas recientes
"""

import numpy as np
from datetime import datetime
from datetime import timezone
from typing import List, Dict, Optional, Tuple


class PerfilUsuario:
    """Construcción de perfiles de usuario multi-fuente"""
    
    LAMBDA_DECAY = 0.000001  # Decay temporal para historial implícito
    EMBEDDING_DIM = 768      # Dimensión de embeddings
    
    # Pesos para combinación lineal
    ALPHA = 0.4  # Peso historial implícito
    BETA = 0.3   # Peso preguntas recientes
    GAMMA = 0.3  # Peso feedback positivo
    DELTA = 0.2  # Peso feedback negativo
    
    @staticmethod
    def perfil_implicito(historial: List[Dict]) -> Optional[np.ndarray]:
        """
        Construye perfil implícito con decay temporal.
        
        Args:
            historial: Lista de items con material_embedding y fecha_consulta
            
        Returns:
            Vector promedio ponderado por decay temporal o None
        """
        if not historial:
            return None
        
        ahora = datetime.now(timezone.utc)
        
        vectores = []
        pesos = []
        
        for item in historial:
            emb = item.get("material_embedding")
            fecha = item.get("fecha_consulta")
            
            if emb is None or fecha is None:
                continue
            
            # Convertir a numpy array
            d = np.array(emb)

            if fecha.tzinfo is None:
             # Si la fecha no tiene timezone, asumirla como UTC
                fecha = fecha.replace(tzinfo=timezone.utc)
            
            # Calcular decay temporal
            delta = (ahora - fecha).total_seconds()
            w = np.exp(-PerfilUsuario.LAMBDA_DECAY * delta)
            
            vectores.append(d)
            pesos.append(w)
        
        if not vectores:
            return None
        
        vectores = np.array(vectores)
        pesos = np.array(pesos)
        
        # Promedio ponderado
        u = np.sum(vectores * pesos[:, None], axis=0) / np.sum(pesos)
        
        return u
    
    @staticmethod
    def perfil_explicito(materiales: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Construye perfiles positivo y negativo basados en ratings.
        
        Args:
            materiales: Lista con material_embedding y calificacion
            
        Returns:
            Tupla (u_pos, u_neg) con vectores promedio
        """
        positivos = []
        negativos = []
        
        for item in materiales:
            emb = item.get("material_embedding")
            rating = item.get("calificacion")
            
            if emb is None or rating is None:
                continue
            
            d = np.array(emb)
            
            # Clasificar según rating
            if rating >= 4:
                positivos.append(d)
            elif rating <= 2:
                negativos.append(d)
        
        u_pos = np.mean(positivos, axis=0) if positivos else None
        u_neg = np.mean(negativos, axis=0) if negativos else None
        
        return u_pos, u_neg
    
    @staticmethod
    def perfil_reciente(preguntas: List[Dict]) -> Optional[np.ndarray]:
        """
        Construye perfil de intención reciente.
        
        Args:
            preguntas: Lista con pregunta_embedding
            
        Returns:
            Vector promedio de preguntas recientes o None
        """
        vectores = [
            np.array(p["pregunta_embedding"])
            for p in preguntas
            if p.get("pregunta_embedding") is not None
        ]
        
        if not vectores:
            return None
        
        return np.mean(vectores, axis=0)
    
    @staticmethod
    def combinar_perfiles(
        u_imp: Optional[np.ndarray],
        u_rec: Optional[np.ndarray],
        u_pos: Optional[np.ndarray],
        u_neg: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Combina perfiles con pesos configurables.
        
        Fórmula: u = α*u_implicit + β*u_recent + γ*u_pos - δ*u_neg
        
        Args:
            u_imp: Perfil implícito
            u_rec: Perfil reciente
            u_pos: Perfil positivo
            u_neg: Perfil negativo
            
        Returns:
            Vector de perfil combinado normalizado
        """
        u = np.zeros(PerfilUsuario.EMBEDDING_DIM)
        
        if u_imp is not None:
            u += PerfilUsuario.ALPHA * u_imp
        
        if u_rec is not None:
            u += PerfilUsuario.BETA * u_rec
        
        if u_pos is not None:
            u += PerfilUsuario.GAMMA * u_pos
        
        if u_neg is not None:
            u -= PerfilUsuario.DELTA * u_neg
        
        # Normalizar (L2 norm)
        norm = np.linalg.norm(u)
        if norm > 0:
            u = u / norm
        
        return u
    
    @classmethod
    def construir_perfil_completo(
        cls,
        historial_implicito: List[Dict],
        feedback_explicito: List[Dict],
        preguntas_recientes: List[Dict]
    ) -> Dict:
        """
        Función principal que construye el perfil completo del usuario.
        
        Args:
            historial_implicito: Datos de biblioteca
            feedback_explicito: Datos de ratings
            preguntas_recientes: Últimas preguntas
            
        Returns:
            Dict con perfil final y componentes intermedios
        """
        # Construir perfiles individuales
        u_implicit = cls.perfil_implicito(historial_implicito)
        u_recent = cls.perfil_reciente(preguntas_recientes)
        u_pos, u_neg = cls.perfil_explicito(feedback_explicito)
        
        # Combinar perfiles
        u_final = cls.combinar_perfiles(u_implicit, u_recent, u_pos, u_neg)
        
        return {
            "perfil_final": u_final.tolist(),
            "componentes": {
                "implicito": u_implicit.tolist() if u_implicit is not None else None,
                "reciente": u_recent.tolist() if u_recent is not None else None,
                "positivo": u_pos.tolist() if u_pos is not None else None,
                "negativo": u_neg.tolist() if u_neg is not None else None
            },
            "metadata": {
                "items_implicitos": len(historial_implicito),
                "items_explicitos": len(feedback_explicito),
                "preguntas": len(preguntas_recientes),
                "dimension": PerfilUsuario.EMBEDDING_DIM
            }
        }