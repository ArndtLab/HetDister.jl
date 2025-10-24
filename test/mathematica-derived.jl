

# the foloowing function are computed by mathematica

hidm(L,N0,mu,r) = (32*L*mu^2*N0^2)/(1 + 4*mu*N0*r)^3


hidm_doww(L,N0,T1,N1,mu,r) = 16*L*mu^2*((2*N1^2)/(1 + 4*mu*N1*r)^3 + ((2*N0^2)/(1 + 4*mu*N0*r)^3 - (2*N1^2)/(1 + 4*mu*N1*r)^3 + (N0*T1)/(1 + 4*mu*N0*r)^2 - (N1*T1)/(1 + 4*mu*N1*r)^2 + (mu*(-N0 + N1)*r*T1^2)/((1 + 4*mu*N0*r)*(1 + 4*mu*N1*r)))/exp(((N1^(-1) + 4*mu*r)*T1)/2))
hidm_down_up(L,N0,T1,N1,T2,N2,mu,r) = (2*L*mu^2*((-16*(-1 + exp(((-N2^(-1) - 4*mu*r)*T2)/2))*N2^4)/(1 + 4*mu*N2*r)^3 + (exp(((-N2^(-1) - 4*mu*r)*T2)/2)*(N1^2 + (-1 + 2*N1)*N2^2)*((8*N1^2)/(1 + 4*mu*N1*r)^3 + ((8*N0^2)/(1 + 4*mu*N0*r)^3 - (8*N1^2)/(1 + 4*mu*N1*r)^3 + (4*N0*T1)/(1 + 4*mu*N0*r)^2 - (4*N1*T1)/(1 + 4*mu*N1*r)^2 + (4*mu*(-N0 + N1)*r*T1^2)/((1 + 4*mu*N0*r)*(1 + 4*mu*N1*r)))/exp(((N1^(-1) + 4*mu*r)*T1)/2)))/N1 - (8*exp(((-N2^(-1) - 4*mu*r)*T2)/2)*N2^3*T2)/(1 + 4*mu*N2*r)^2 + (4*exp(-1/2*((N1^(-1) + 4*mu*r)*T1) - ((N2^(-1) + 4*mu*r)*T2)/2)*(N1^2 + (-1 + 2*N1)*N2^2)*(exp(((N1^(-1) + 4*mu*r)*T1)/2)*N1*(1 + 4*mu*N0*r)^2 + (N0 - N1)*(1 + 2*mu*r*(-8*mu*N0*N1*r - (1 + 4*mu*N0*r)*(1 + 4*mu*N1*r)*T1)))*T2)/(N1*(1 + 4*mu*N0*r)^2*(1 + 4*mu*N1*r)^2) - (2*exp(((-N2^(-1) - 4*mu*r)*T2)/2)*N2^2*T2^2)/(1 + 4*mu*N2*r) + exp(((-N2^(-1) - 4*mu*r)*T2)/2)*(N1*(-1 + N2^2/N1^2) + ((N1^2 + (-1 + 2*N1)*N2^2)*((1 + 4*mu*N1*r)^(-1) + ((1 + 4*mu*N0*r)^(-1) + (-1 - 4*mu*N1*r)^(-1))/exp(((N1^(-1) + 4*mu*r)*T1)/2)))/N1)*T2^2))/N2^2

hidm_up(L,N0,T1,N1,mu,r) = 4*L*mu^2*((8*N1^2)/(1 + 4*mu*N1*r)^3 + exp(((-N1^(-1) - 4*mu*r)*T1)/2)*((4*N0*(N0^2 + (-1 + 2*N0)*N1^2))/(N1^2*(1 + 4*mu*N0*r)^3) - (8*N1^2)/(1 + 4*mu*N1*r)^3 + (2*(N0^2 + (-1 + 2*N0)*N1^2)*T1)/(N1^2*(1 + 4*mu*N0*r)^2) - (4*N1*T1)/(1 + 4*mu*N1*r)^2 - T1^2/(1 + 4*mu*N1*r) + ((-2*mu*N0^2*r + N1^2*(1 + 2*mu*r))*T1^2)/(N1^2*(1 + 4*mu*N0*r))))
hidm_up_down(L,N0,T1,N1,T2,N2,mu,r) = (L*((-64*(-1 + exp(((-N2^(-1) - 4*mu*r)*T2)/2))*mu^2*N2^2)/(1 + 4*mu*N2*r)^3 + (4*exp(((-N2^(-1) - 4*mu*r)*T2)/2)*mu^2*((16*N1^4)/(1 + 4*mu*N1*r)^3 + exp(((-N1^(-1) - 4*mu*r)*T1)/2)*((8*N0*(N0^2 + (-1 + 2*N0)*N1^2))/(1 + 4*mu*N0*r)^3 - (16*N1^4)/(1 + 4*mu*N1*r)^3 + (4*(N0^2 + (-1 + 2*N0)*N1^2)*T1)/(1 + 4*mu*N0*r)^2 - (8*N1^3*T1)/(1 + 4*mu*N1*r)^2 - (2*N1^2*T1^2)/(1 + 4*mu*N1*r) + (2*(-2*mu*N0^2*r + N1^2*(1 + 2*mu*r))*T1^2)/(1 + 4*mu*N0*r))))/N1^2 - (32*mu^2*N2*T2)/(exp(((N2^(-1) + 4*mu*r)*T2)/2)*(1 + 4*mu*N2*r)^2) - (8*exp(((-N2^(-1) - 4*mu*r)*T2)/2)*mu^2*((-2*N0^2 + 2*(1 - 2*N0)*N1^2)/(exp(((N1^(-1) + 4*mu*r)*T1)/2)*(1 + 4*mu*N0*r)^2) + (4*(-1 + exp(((-N1^(-1) - 4*mu*r)*T1)/2))*N1^3)/(1 + 4*mu*N1*r)^2 + (2*N1^2*T1)/(exp(((N1^(-1) + 4*mu*r)*T1)/2)*(1 + 4*mu*N1*r)) + (2*(2*mu*N0^2*r - N1^2*(1 + 2*mu*r))*T1)/(exp(((N1^(-1) + 4*mu*r)*T1)/2)*(1 + 4*mu*N0*r)))*T2)/N1^2 - (8*mu^2*T2^2)/(exp(((N2^(-1) + 4*mu*r)*T2)/2)*(1 + 4*mu*N2*r)) + (8*exp(-1/2*((N1^(-1) + 4*mu*r)*T1) - ((N2^(-1) + 4*mu*r)*T2)/2)*mu^2*(exp(((N1^(-1) + 4*mu*r)*T1)/2)*N1^2*(1 + 4*mu*N0*r) + 2*mu*(-N0 + N1)*r*(N1 + N1^2*(2 + 4*mu*r) + N0*(1 + 4*mu*N1*r)))*T2^2)/(N1^2*(1 + 4*mu*N0*r)*(1 + 4*mu*N1*r))))/2





hidm(L,N0,T1,N1,mu,r) = N0 < N1 ?  hidm_up(L,N0,T1,N1,mu,r) : hidm_doww(L,N0,T1,N1,mu,r)

function hidm(L,N0,T1,N1,T2,N2,mu,r)
	@assert T1 >=0 && T2 >=0 && N0 > 0 && N1 > 0 && N2 > 0 && mu > 0 && r > 0
	if N0 < N1
		if N1 >= N2
			return hidm_up_down(L,N0,T1,N1,T2,N2,mu,r)
		else
			throw(ArgumentError("not implemented for such Ns"))
			return nothing
		end
	else
		if N1 <= N2
			return hidm_down_up(L,N0,T1,N1,T2,N2,mu,r)
		else
			throw(ArgumentError("not implemented for such Ns"))
			return nothing
		end
	end
end

