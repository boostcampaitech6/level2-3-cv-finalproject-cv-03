import React from 'react';
import { View, Text } from 'react-native';

interface Props {
  navigation: any;
}

const Tab1Screen: React.FC<Props> = ({ navigation }) => {
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Text>Tab 1 Screen</Text>
    </View>
  );
};

export default Tab1Screen;