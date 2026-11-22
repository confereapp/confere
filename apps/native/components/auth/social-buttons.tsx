import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { useToast } from "heroui-native";
import { useState } from "react";
import { ActivityIndicator, Pressable, Text, View } from "react-native";
import { authClient } from "@/lib/auth/client";
import { NETWORK_ERROR_MESSAGE } from "@/lib/auth/error-messages";
import { trpc } from "@/lib/trpc";

const GOOGLE_COLORS = ["#EA4335", "#4285F4", "#FBBC05", "#34A853"];

// Só aparece quando o servidor tem GOOGLE_CLIENT_ID e GOOGLE_CLIENT_SECRET.
export function SocialButtons() {
  const router = useRouter();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { data } = useQuery(trpc.authProviders.queryOptions());

  if (!data?.google) return null;

  async function signInWithGoogle() {
    if (isSubmitting) return;
    setIsSubmitting(true);
    try {
      // "/" vira o deep link confere:// e a tela inicial redireciona conforme a sessão.
      const { error } = await authClient.signIn.social({
        provider: "google",
        callbackURL: "/",
      });
      if (error) {
        toast.show({
          variant: "danger",
          label: "Não foi possível entrar com Google. Tente novamente.",
        });
        return;
      }
      // Navegador fechado sem login não gera erro: só segue se a sessão existir.
      const session = await authClient.getSession();
      if (session.data) router.replace("/(tabs)");
    } catch {
      toast.show({ variant: "danger", label: NETWORK_ERROR_MESSAGE });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <View className="flex-row items-center gap-3">
        <View className="h-px flex-1 bg-hair" />
        <Text className="font-instrument text-ink3 text-xs">ou</Text>
        <View className="h-px flex-1 bg-hair" />
      </View>

      <Pressable
        className="h-[52px] flex-row items-center justify-center gap-2.5 rounded-[26px] border-[0.5px] border-glass-bd bg-glass shadow-glass backdrop-blur-[18px] backdrop-saturate-[180%] active:opacity-80"
        accessibilityRole="button"
        disabled={isSubmitting}
        onPress={signInWithGoogle}
      >
        {isSubmitting ? (
          <ActivityIndicator />
        ) : (
          <>
            <View className="size-[18px] flex-row flex-wrap overflow-hidden rounded-full">
              {GOOGLE_COLORS.map((color) => (
                <View
                  key={color}
                  className="size-[9px]"
                  style={{ backgroundColor: color }}
                />
              ))}
            </View>
            <Text className="font-instrument-semibold text-ink text-sm">
              Continuar com Google
            </Text>
          </>
        )}
      </Pressable>
    </>
  );
}
