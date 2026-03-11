"""
Endpoints para el Sistema de Recomendaciones
Construye perfiles de usuario y genera recomendaciones personalizadas
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.database import SessionLocal
from src.services.recomendador import PerfilUsuario
from src.routers.auth import get_current_user

# Importar modelos necesarios
from src.models.biblioteca import Biblioteca
from src.models.material_estudio import MaterialEstudio
from src.models.mensaje_pregunta import MensajePregunta
from src.models.mensaje_respuesta import MensajeRespuesta
from src.models.respuesta_material import RespuestaMaterial
from src.models.mensaje import Mensaje
from src.models.chat import Chat

from sqlalchemy import text
from typing import Optional
router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/perfil")
def construir_perfil_usuario(
    limit_historial: int = 50,
    limit_preguntas: int = 10,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Construye el perfil completo del usuario autenticado combinando múltiples fuentes.
    
    Args:
        limit_historial: Límite de items del historial implícito (default: 50)
        limit_preguntas: Límite de preguntas recientes (default: 10)
        current_user: Usuario autenticado
        db: Sesión de base de datos
    
    Returns:
        Perfil de usuario con vector final y componentes individuales
    """
    
    username = current_user.usuario
    
    # 1️⃣ Obtener historial implícito (biblioteca)
    resultados_historial = (
        db.query(
            Biblioteca.material_id,
            Biblioteca.documento_id,
            Biblioteca.fecha_consulta,
            MaterialEstudio.material_embedding
        )
        .outerjoin(MaterialEstudio, MaterialEstudio.id == Biblioteca.material_id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .limit(limit_historial)
        .all()
    )
    
    historial = [
        {
            "tipo": "fragmento" if material_id else "documento_completo",
            "material_id": material_id,
            "documento_id": documento_id,
            "fecha_consulta": fecha_consulta,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, documento_id, fecha_consulta, material_embedding in resultados_historial
    ]
    
    # 2️⃣ Obtener feedback explícito (ratings)
    resultados_feedback = (
        db.query(
            RespuestaMaterial.material_id,
            MensajeRespuesta.calificacion,
            MaterialEstudio.material_embedding
        )
        .join(MensajeRespuesta, RespuestaMaterial.mensaje_respuesta_id == MensajeRespuesta.mensaje_id)
        .join(MaterialEstudio, MaterialEstudio.id == RespuestaMaterial.material_id)
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion.isnot(None)
        )
        .all()
    )
    
    feedback = [
        {
            "material_id": material_id,
            "calificacion": calificacion,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, calificacion, material_embedding in resultados_feedback
    ]
    
    # Estadísticas de feedback
    total_respuestas = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .scalar()
    )
    
    calificadas_5 = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion == 5
        )
        .scalar()
    )
    
    calificadas_1 = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion == 1
        )
        .scalar()
    )
    
    calificadas_null = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion.is_(None)
        )
        .scalar()
    )
    
    # 3️⃣ Obtener preguntas recientes
    resultados_preguntas = (
        db.query(MensajePregunta.pregunta_embedding)
        .join(Mensaje, Mensaje.id == MensajePregunta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .order_by(Mensaje.fecha_mensaje.desc())
        .limit(limit_preguntas)
        .all()
    )
    
    preguntas = [
        {
            "pregunta_embedding": [float(x) for x in pregunta_embedding[0]] if pregunta_embedding[0] is not None else None
        }
        for pregunta_embedding in resultados_preguntas
    ]
    
    # 4️⃣ Construir perfil usando el servicio
    try:
        perfil = PerfilUsuario.construir_perfil_completo(
            historial_implicito=historial,
            feedback_explicito=feedback,
            preguntas_recientes=preguntas
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error construyendo perfil de usuario: {str(e)}"
        )
    
    # 5️⃣ Retornar perfil completo
    return {
        "username": username,
        **perfil,
        "estadisticas_feedback": {
            "total_respuestas": total_respuestas,
            "calificadas_5": calificadas_5,
            "calificadas_1": calificadas_1,
            "calificadas_null": calificadas_null
        }
    }


@router.get("/perfil/componentes")
def obtener_componentes_perfil(
    limit_historial: int = 50,
    limit_preguntas: int = 10,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Obtiene solo los componentes individuales del perfil sin combinarlos.
    Útil para debugging o análisis detallado.
    """
    
    username = current_user.usuario
    
    # Obtener datos usando las mismas queries
    # 1. Historial
    resultados_historial = (
        db.query(
            Biblioteca.material_id,
            Biblioteca.documento_id,
            Biblioteca.fecha_consulta,
            MaterialEstudio.material_embedding
        )
        .outerjoin(MaterialEstudio, MaterialEstudio.id == Biblioteca.material_id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .limit(limit_historial)
        .all()
    )
    
    historial = [
        {
            "tipo": "fragmento" if material_id else "documento_completo",
            "material_id": material_id,
            "documento_id": documento_id,
            "fecha_consulta": fecha_consulta,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, documento_id, fecha_consulta, material_embedding in resultados_historial
    ]
    
    # 2. Feedback
    resultados_feedback = (
        db.query(
            RespuestaMaterial.material_id,
            MensajeRespuesta.calificacion,
            MaterialEstudio.material_embedding
        )
        .join(MensajeRespuesta, RespuestaMaterial.mensaje_respuesta_id == MensajeRespuesta.mensaje_id)
        .join(MaterialEstudio, MaterialEstudio.id == RespuestaMaterial.material_id)
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion.isnot(None)
        )
        .all()
    )
    
    feedback = [
        {
            "material_id": material_id,
            "calificacion": calificacion,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, calificacion, material_embedding in resultados_feedback
    ]
    
    # 3. Preguntas
    resultados_preguntas = (
        db.query(MensajePregunta.pregunta_embedding)
        .join(Mensaje, Mensaje.id == MensajePregunta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .order_by(Mensaje.fecha_mensaje.desc())
        .limit(limit_preguntas)
        .all()
    )
    
    preguntas = [
        {
            "pregunta_embedding": [float(x) for x in pregunta_embedding[0]] if pregunta_embedding[0] is not None else None
        }
        for pregunta_embedding in resultados_preguntas
    ]
    
    # Calcular componentes individuales
    u_implicit = PerfilUsuario.perfil_implicito(historial)
    u_recent = PerfilUsuario.perfil_reciente(preguntas)
    u_pos, u_neg = PerfilUsuario.perfil_explicito(feedback)
    
    return {
        "username": username,
        "componentes": {
            "implicito": {
                "vector": u_implicit.tolist() if u_implicit is not None else None,
                "num_items": len(historial)
            },
            "reciente": {
                "vector": u_recent.tolist() if u_recent is not None else None,
                "num_preguntas": len(preguntas)
            },
            "positivo": {
                "vector": u_pos.tolist() if u_pos is not None else None,
                "num_items": len([m for m in feedback if m.get("calificacion") == 5])
            },
            "negativo": {
                "vector": u_neg.tolist() if u_neg is not None else None,
                "num_items": len([m for m in feedback if m.get("calificacion") == 1])
            }
        },
        "pesos_usados": {
            "alpha_implicito": PerfilUsuario.ALPHA,
            "beta_reciente": PerfilUsuario.BETA,
            "gamma_positivo": PerfilUsuario.GAMMA,
            "delta_negativo": PerfilUsuario.DELTA
        }
    }


@router.get("/perfil/estadisticas")
def obtener_estadisticas_perfil(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Obtiene estadísticas resumidas del perfil del usuario.
    No calcula embeddings, solo cuenta registros.
    """
    
    username = current_user.usuario
    
    # Contar historial
    total_historial = (
        db.query(func.count(Biblioteca.id))
        .filter(Biblioteca.usuario == username)
        .scalar()
    )
    
    # Contar respuestas y calificaciones
    total_respuestas = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .scalar()
    )
    
    calificadas_5 = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username, MensajeRespuesta.calificacion == 5)
        .scalar()
    )
    
    calificadas_1 = (
        db.query(func.count(MensajeRespuesta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username, MensajeRespuesta.calificacion == 1)
        .scalar()
    )
    
    # Contar preguntas
    total_preguntas = (
        db.query(func.count(MensajePregunta.mensaje_id))
        .join(Mensaje, Mensaje.id == MensajePregunta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .scalar()
    )
    
    return {
        "username": username,
        "historial_implicito": total_historial,
        "feedback_total": total_respuestas,
        "feedback_positivo": calificadas_5,
        "feedback_negativo": calificadas_1,
        "feedback_sin_calificar": total_respuestas - calificadas_5 - calificadas_1,
        "preguntas_totales": total_preguntas
    }

@router.get("/materiales-por-componente")
def recomendar_por_componente(
    top_k: int = 3,
    fuente: Optional[str] = None,
    nivel_dificultad: Optional[str] = None,
    limit_historial: int = 50,
    limit_preguntas: int = 10,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Genera recomendaciones basadas en CADA componente del perfil por separado.
    Retorna top_k materiales más similares para: perfil_final, implícito, reciente y positivo.
    
    Args:
        top_k: Número de recomendaciones por componente (default: 3)
        fuente: Filtrar por fuente (CODIGO, PDF, VIDEO, GIT) - opcional
        nivel_dificultad: Filtrar por nivel (Baja, Media, Alta) - opcional
        limit_historial: Items de historial para construir perfil
        limit_preguntas: Preguntas recientes para construir perfil
        current_user: Usuario autenticado
        db: Sesión de base de datos
    
    Returns:
        Recomendaciones separadas por componente (final, implícito, reciente, positivo)
    """
    
    username = current_user.usuario
    
    # 1️⃣ Construir perfil del usuario (MISMA LÓGICA que /perfil)
    # Historial
    resultados_historial = (
        db.query(
            Biblioteca.material_id,
            Biblioteca.documento_id,
            Biblioteca.fecha_consulta,
            MaterialEstudio.material_embedding
        )
        .outerjoin(MaterialEstudio, MaterialEstudio.id == Biblioteca.material_id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .limit(limit_historial)
        .all()
    )
    
    historial = [
        {
            "tipo": "fragmento" if material_id else "documento_completo",
            "material_id": material_id,
            "documento_id": documento_id,
            "fecha_consulta": fecha_consulta,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, documento_id, fecha_consulta, material_embedding in resultados_historial
    ]
    
    # Feedback
    resultados_feedback = (
        db.query(
            RespuestaMaterial.material_id,
            MensajeRespuesta.calificacion,
            MaterialEstudio.material_embedding
        )
        .join(MensajeRespuesta, RespuestaMaterial.mensaje_respuesta_id == MensajeRespuesta.mensaje_id)
        .join(MaterialEstudio, MaterialEstudio.id == RespuestaMaterial.material_id)
        .join(Mensaje, Mensaje.id == MensajeRespuesta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(
            Chat.usuario == username,
            MensajeRespuesta.calificacion.isnot(None)
        )
        .all()
    )
    
    feedback = [
        {
            "material_id": material_id,
            "calificacion": calificacion,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, calificacion, material_embedding in resultados_feedback
    ]
    
    # Preguntas
    resultados_preguntas = (
        db.query(MensajePregunta.pregunta_embedding)
        .join(Mensaje, Mensaje.id == MensajePregunta.mensaje_id)
        .join(Chat, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .order_by(Mensaje.fecha_mensaje.desc())
        .limit(limit_preguntas)
        .all()
    )
    
    preguntas = [
        {
            "pregunta_embedding": [float(x) for x in pregunta_embedding[0]] if pregunta_embedding[0] is not None else None
        }
        for pregunta_embedding in resultados_preguntas
    ]
    
    # 2️⃣ Calcular componentes individuales
    u_implicit = PerfilUsuario.perfil_implicito(historial)
    u_recent = PerfilUsuario.perfil_reciente(preguntas)
    u_pos, u_neg = PerfilUsuario.perfil_explicito(feedback)
    u_final = PerfilUsuario.combinar_perfiles(u_implicit, u_recent, u_pos, u_neg)
    
    perfil_vacio = (
        u_implicit is None and 
        u_recent is None and 
        u_pos is None
    )
    
    if perfil_vacio:
        # ⭐ ESTRATEGIA COLD START: Recomendar lo más popular
        # return recomendar_cold_start(
        #     username=username,
        #     top_k=top_k,
        #     fuente=fuente,
        #     nivel_dificultad=nivel_dificultad,
        #     db=db
        # )
        return {"modo": "cold_start"}
    
    # 3️⃣ Función auxiliar para buscar top_k similares
    def buscar_top_k(vector, nombre_componente):
        """
        Busca los top_k materiales más similares a un vector dado.
        Usa operador <=> de pgvector (distancia coseno).
        """
        if vector is None:
            return {
                "componente": nombre_componente,
                "disponible": False,
                "razon": "No hay datos suficientes para este componente",
                "materiales": []
            }
        
        # Convertir numpy array a lista
        vector_lista = vector.tolist()
        vector_str = str(vector_lista) 

         # 🆕 OBTENER MATERIALES YA VISTOS POR EL USUARIO
        materiales_vistos = db.query(Biblioteca.material_id).filter(
            Biblioteca.usuario == username,
            Biblioteca.material_id.isnot(None)  # Solo fragmentos, no documentos completos
        ).distinct().limit(50).all()
        
        ids_vistos = [m[0] for m in materiales_vistos]
        
        # Query SQL con pgvector
        sql_query = """
            SELECT 
                me.id,
                d.id as documento_id,
                d.nombre_documento,
                d.fuente,
                d.url,
                d.tematica,
                d.competencia,
                d.nivel_dificultad,
                (me.material_embedding <=> :perfil_vector) AS distancia
            FROM material_estudio me
            JOIN documento d ON d.id = me.documento_id
            WHERE 1=1
        """
        
        params = {
            "perfil_vector": vector_str,  # ← Agregar el vector aquí
            "limit": top_k
        }

        # 🆕 EXCLUIR MATERIALES YA VISTOS
        if ids_vistos:
            # Convertir lista a formato PostgreSQL: ANY(ARRAY[1,2,3])
            sql_query += " AND me.id != ALL(:ids_vistos)"
            params["ids_vistos"] = ids_vistos
        
        # Filtros opcionales
        if fuente:
            sql_query += " AND d.fuente = :fuente"
            params["fuente"] = fuente.upper()
        
        if nivel_dificultad:
            sql_query += " AND d.nivel_dificultad = :nivel"
            params["nivel"] = nivel_dificultad
        
        sql_query += """
            ORDER BY distancia ASC
            LIMIT :limit
        """
        
        try:
            resultados = db.execute(text(sql_query), params).fetchall()
            
            materiales = []
            for row in resultados:
                # Convertir distancia coseno a similitud (1 - distancia)
                similitud = 1 - float(row.distancia)
                
                materiales.append({
                    "material_id": row.id,
                    "documento_id": row.documento_id,
                    "nombre_documento": row.nombre_documento,
                    "fuente": row.fuente,
                    "url": row.url,
                    "tematica": row.tematica,
                    "competencia": row.competencia,
                    "nivel_dificultad": row.nivel_dificultad,
                    "distancia_coseno": float(row.distancia),
                    "similitud": similitud,
                    "score_porcentaje": round(similitud * 100, 2),
                    "es_nuevo": True # 🆕 Marcar que NO ha sido visto
                })
            
            return {
                "componente": nombre_componente,
                "disponible": True,
                "total_materiales": len(materiales),
                "materiales_excluidos": len(ids_vistos), 
                "materiales": materiales
            }
            
        except Exception as e:
            return {
                "componente": nombre_componente,
                "disponible": False,
                "error": str(e),
                "materiales": []
            }
    
    # 4️⃣ Buscar top_k para cada componente
    recomendaciones_final = buscar_top_k(u_final, "perfil_final")
    recomendaciones_implicito = buscar_top_k(u_implicit, "implicito")
    recomendaciones_reciente = buscar_top_k(u_recent, "reciente")
    recomendaciones_positivo = buscar_top_k(u_pos, "positivo")
    
    # 5️⃣ Retornar resultado estructurado
    return {
        "username": username,
        "top_k": top_k,
        "filtros_aplicados": {
            "fuente": fuente,
            "nivel_dificultad": nivel_dificultad
        },
        "estadisticas_perfil": {
            "items_implicitos": len(historial),
            "items_explicitos": len(feedback),
            "preguntas": len(preguntas),
            "positivos": len([m for m in feedback if m.get("calificacion") == 5]),
            "negativos": len([m for m in feedback if m.get("calificacion") == 1])
        },
        "recomendaciones_por_componente": {
            "perfil_final": recomendaciones_final,
            "implicito": recomendaciones_implicito,
            "reciente": recomendaciones_reciente,
            "positivo": recomendaciones_positivo
        }
    }

def recomendar_cold_start(
    username: str,
    top_k: int,
    fuente: Optional[str],
    nivel_dificultad: Optional[str],
    db: Session
):
    """
    Estrategia para usuarios nuevos sin historial.
    Recomienda materiales más populares (más vistos/calificados).
    """
    
    # Query para encontrar materiales más populares
    sql_query = """
        SELECT 
            me.id,
            d.id as documento_id,
            d.nombre_documento,
            d.fuente,
            d.url,
            d.tematica,
            d.competencia,
            d.nivel_dificultad,
            COUNT(DISTINCT b.usuario) as popularidad_biblioteca,
            COUNT(DISTINCT rm.mensaje_respuesta_id) as popularidad_respuestas,
            AVG(mr.calificacion) as rating_promedio
        FROM material_estudio me
        JOIN documento d ON d.id = me.documento_id
        LEFT JOIN biblioteca b ON b.material_id = me.id
        LEFT JOIN respuesta_material rm ON rm.material_id = me.id
        LEFT JOIN mensaje_respuesta mr ON mr.mensaje_id = rm.mensaje_respuesta_id
        WHERE 1=1
    """
    
    params = {"limit": top_k}
    
    if fuente:
        sql_query += " AND d.fuente = :fuente"
        params["fuente"] = fuente.upper()
    
    if nivel_dificultad:
        sql_query += " AND d.nivel_dificultad = :nivel"
        params["nivel"] = nivel_dificultad
    
    sql_query += """
        GROUP BY me.id, d.id
        ORDER BY 
            popularidad_biblioteca DESC,
            popularidad_respuestas DESC,
            rating_promedio DESC NULLS LAST
        LIMIT :limit
    """
    
    resultados = db.execute(text(sql_query), params).fetchall()
    
    materiales = []
    for row in resultados:
        materiales.append({
            "material_id": row.id,
            "documento_id": row.documento_id,
            "nombre_documento": row.nombre_documento,
            "fuente": row.fuente,
            "url": row.url,
            "tematica": row.tematica,
            "competencia": row.competencia,
            "nivel_dificultad": row.nivel_dificultad,
            "popularidad": row.popularidad_biblioteca,
            "rating_promedio": float(row.rating_promedio) if row.rating_promedio else None,
            "razon": "Material popular (usuario nuevo)"
        })
    
    return {
        "username": username,
        "modo": "cold_start",
        "razon": "Usuario sin historial suficiente - mostrando contenido popular",
        "top_k": top_k,
        "filtros_aplicados": {
            "fuente": fuente,
            "nivel_dificultad": nivel_dificultad
        },
        "recomendaciones": materiales
    }