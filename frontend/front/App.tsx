import React, { useState } from "react";
import { Text } from "react-native";
import { useFonts } from "expo-font";
import { NavigationContainer } from "@react-navigation/native";
import RootStackNavigator from "./src/navigation/RootStackNavigator";
import { ThemeProvider, createTheme } from "@rneui/themed";
import { UserContext } from "./src/UserContext";

const theme = createTheme({
  lightColors: {
    primary: "#FFA000",
  },
  mode: "light",
  components: {
    Button: {
      radius: "10",
      style: {
        width: 200,
        marginVertical: 5,
      },
    },
    Text: {
      style: {
        fontFamily: "NGB",
      },
    },
  },
});

export default function App() {
  const [user, setUser] = useState(null);
  const [fontsLoaded] = useFonts({
    SG: require("./assets/fonts/SEBANG Gothic.ttf"),
    NGB: require("./assets/fonts/NanumGothicBold.otf"),
    SGB: require("./assets/fonts/SEBANG Gothic Bold.ttf"),
    KUD: require("./assets/fonts/KoddiUDOnGothic-Regular.ttf"),
    C24: require("./assets/fonts/Cafe24Ssurround-v2.0.ttf"),
  });

  if (!fontsLoaded) {
    return <Text style={{ fontSize: 20, marginTop: 50 }}>Loading...</Text>;
  }

  return (
    <UserContext.Provider value={{ user, setUser }}>
      <ThemeProvider theme={theme}>
        <NavigationContainer>
          <RootStackNavigator />
        </NavigationContainer>
      </ThemeProvider>
    </UserContext.Provider>
  );
}
