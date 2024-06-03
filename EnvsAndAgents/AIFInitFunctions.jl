using ActiveInference

function init_agent_lr_alpha(lr_AIF, alpha_AIF)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_AIF,
                                            "alpha" => alpha_AIF)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end


function init_AIF_agent_full_eksA(beta, lr_pB, fr_pB)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_pB,
                                        "fr_pB" => fr_pB)
    C = array_of_any_zeros(4)
    C[1][1] = 3.0 # CC
    C[1][2] = 1.0 # CD
    C[1][3] = 4.0 # DC
    C[1][4] = 2.0 # DD

    β = beta
    C[1] = softmax(C[1] * β)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end

function init_AIF_agent_full(alpha, beta, lr_pB, fr_pB)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_pB,
                                        "fr_pB" => fr_pB,
                                        "alpha" => alpha)

    C = array_of_any_zeros(4)
    C[1][1] = 3.0 # CC
    C[1][2] = 1.0 # CD
    C[1][3] = 4.0 # DC
    C[1][4] = 2.0 # DD

    β = beta
    C[1] = softmax(C[1] * β)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end

function init_AIF_agent_beta_lr_pB(beta, lr_pB)
    parameters_AIF = Dict{String, Real}(
        "lr_pB" => lr_pB
    )

    C = array_of_any_zeros(4)
    C[1][1] = 3.0 # CC
    C[1][2] = 1.0 # CD
    C[1][3] = 4.0 # DC
    C[1][4] = 2.0 # DD

    β = beta
    C[1] = softmax(C[1] * β)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end

function init_AIF_agents(lr_samuel, lr_jonathan)
    parameters_samuel = Dict{String, Real}("lr_pB" => lr_samuel,
                                            "alpha" => 16)
    parameters_jonathan = Dict{String, Real}("lr_pB" => lr_jonathan,
                                            "alpha" => 16)

    SAMUEL = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_samuel,
                      verbose = false)

    JONATHAN = init_aif(A_matrix, B_matrix;
                        C=C,
                        pB=pB,
                        settings=settings,
                        parameters=parameters_jonathan,
                        verbose = false)

    return SAMUEL, JONATHAN
end

function init_AIF_agent_lr(lr_AIF)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_AIF,
                                            "alpha" => 16)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end


function init_AIF(settings, parameters)
    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters,
                      verbose = false)

    return AIF_agent
end