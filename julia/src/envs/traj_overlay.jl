mutable struct TrajOverlay <: SceneOverlay
    trajectory::Vector{MPCState}
    ego_id::Int
    line_width::Float64
    color::Colorant

    function TrajOverlay(trajectory::Vector{MPCState};
            ego_id::Int=EGO_ID,
            line_width::Float64=0.1, #[m]
            color::Colorant=colorant"blue",
        )

        new(trajectory, ego_id, line_width, color)
    end
end

function AutoViz.render!(rendermodel::RenderModel, overlay::TrajOverlay, scene::Scene, roadway::Any)
    ego = scene[findfirst(overlay.ego_id, scene)]
    ego_s = ego.state.posF.s
    ego_t = ego.state.posF.t
    ego_lane = get_lane(roadway, ego.state)

    for i in 1:length(overlay.trajectory) - 1
        s1 = overlay.trajectory[i]
        s2 = overlay.trajectory[i+1]
        s1F = Frenet(ego_lane, ego_s + s1.x, ego_t + s1.y, 0.0)
        s2F = Frenet(ego_lane, ego_s + s2.x, ego_t + s2.y, 0.0)
        s1G = get_posG(s1F, roadway)
        s2G = get_posG(s2F, roadway)

        if i == 1
            add_instruction!(rendermodel, render_line_segment,
                                (ego.state.posG.x, ego.state.posG.y,
                                    s1G.x, s1G.y,
                                    overlay.color, overlay.line_width))
        end

        add_instruction!(rendermodel, render_line_segment,
                            (s1G.x, s1G.y,
                                s2G.x, s2G.y,
                                overlay.color, overlay.line_width))
    end

    rendermodel
end
