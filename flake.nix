{
  description = "Beyond_Generation dev shell";

  outputs = { self, nixpkgs }: {
    devShell.x86_64-linux = with nixpkgs.legacyPackages.x86_64-linux; mkShell {
      buildInputs = [
        python311
        python311Packages.numpy
        python311Packages.pip
        gdal
      ];
    };
  };
}
