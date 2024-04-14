Shader "Custom/DepthData"
{
    SubShader
    {
        Pass
        {
            ZWrite Off
            ColorMask RGB

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _CameraDepthTexture;

            fixed4 frag (v2f i) : SV_Target
            {
                float rawDepth = tex2D(_CameraDepthTexture, i.uv).r;
                float near = _ProjectionParams.z;
                float far = _ProjectionParams.w;
                float linearDepth = near * far / (far - rawDepth * (far - near));
            
                float maxDepth = 10.0;
            
                if (linearDepth > maxDepth)
                {
                    return 0;
                }
                else
                {
                    return linearDepth / maxDepth;
                }
            }
            ENDCG
        }
    }
}